"""W7/W8：Shadow / Reference Bundle（方案 §9）。

Shadow Bundle（每 dataset×seed 一次，1 个 shadow model）:
    bundles/<dataset>/seed<k>/shadow_bundle.npz + checkpoints/shadows/shadow_seed<k>.pt
    覆盖分区输出：shadow_train / shadow_test / auxiliary / rmia_population

Reference Bundle（每 dataset×seed 一次，4 个 reference model）:
    bundles/<dataset>/seed<k>/reference_bundle.npz + checkpoints/references/ref_<i>_seed<k>.pt
    - ref 按 §4.4 训练：种子 100*seed+i，从 reference_pool 无偏采样 50% 子集
    - 记录 in/out 矩阵与逐 ref 概率矩阵（pool / member_eval / nonmember 三视图）
    - 1/2/4 ref 消融直接对概率矩阵切片即可（§23.5）
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import torch

from . import config as C
from . import data as D
from . import models as MO
from . import training as T
from .caching import compute_outputs

SHADOW_PARTITIONS = ["shadow_train", "shadow_test", "auxiliary", "rmia_population"]


def build_shadow(dataset: str, seed: int, epochs_override: int = None) -> Dict[str, Any]:
    cfg = C.load_config(dataset)
    manifest = C.load_manifest(dataset, seed)
    data = D.load_dataset(dataset)
    parts = D.partitions_from_manifest(data, manifest)
    meta = C.DATASET_META[dataset]
    input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))

    model = MO.build_model(cfg["model"]["arch"], input_dim=input_dim, num_classes=meta["num_classes"])
    X_tr, y_tr = parts["shadow_train"]
    t0 = time.time()
    model, history = T.train_classifier(
        model, X_tr, y_tr, cfg["model"], seed=manifest["derived_seeds"]["shadow_model"],
        X_val=parts["shadow_test"][0], y_val=parts["shadow_test"][1],
        epochs_override=epochs_override, log_prefix=f"[{dataset}/s{seed}/shadow] ",
    )
    elapsed = round(time.time() - t0, 1)

    out: Dict[str, np.ndarray] = {}
    for name in SHADOW_PARTITIONS:
        X, y = parts[name]
        outs = compute_outputs(model, X, y)
        for kind, arr in outs.items():
            out[f"{name}_{kind}"] = arr

    ckpt_dir = C.CHECKPOINT_DIR / "shadows" / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"shadow_seed{seed}.pt")

    bundle_dir = C.BUNDLE_DIR / dataset / f"seed{seed}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_dir / "shadow_bundle.npz",
        git_commit=manifest["generated"]["git_commit"],
        shadow_train_seconds=elapsed,
        **out,
    )
    print(f"[shadow-bundle] {dataset} seed{seed} saved ({elapsed}s)")
    return model


def load_shadow_bundle(dataset: str, seed: int) -> Dict[str, np.ndarray]:
    return dict(np.load(C.BUNDLE_DIR / dataset / f"seed{seed}" / "shadow_bundle.npz"))


def build_references(dataset: str, seed: int, epochs_override: int = None) -> List[Any]:
    cfg = C.load_config(dataset)
    manifest = C.load_manifest(dataset, seed)
    data = D.load_dataset(dataset)
    parts = D.partitions_from_manifest(data, manifest)
    meta = C.DATASET_META[dataset]
    input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))
    num_refs = int(cfg["reference"]["num_models"])
    ratio = float(cfg["reference"]["subset_ratio"])

    pool_X, pool_y = parts["reference_pool"]
    pool_gids = np.asarray(manifest["reference_pool"]["indices"], dtype=np.int64)
    pop_gids = np.asarray(manifest["rmia_population"]["indices"], dtype=np.int64)
    # population 在 pool 内的位置
    pop_pos = np.searchsorted(pool_gids, pop_gids)
    assert np.all(pool_gids[pop_pos] == pop_gids), "population not aligned to pool"

    n_pool = len(pool_X)
    subset_size = int(round(n_pool * ratio))

    in_out = np.zeros((num_refs, n_pool), dtype=np.int8)
    subset_positions = np.zeros((num_refs, subset_size), dtype=np.int64)
    ref_probs_pool = np.zeros((num_refs, n_pool, meta["num_classes"]), dtype=np.float32)
    me_X, me_y = parts["target_member_eval"]
    nm_X, nm_y = parts["target_nonmember"]
    ref_probs_me = np.zeros((num_refs, len(me_X), meta["num_classes"]), dtype=np.float32)
    ref_probs_nm = np.zeros((num_refs, len(nm_X), meta["num_classes"]), dtype=np.float32)
    timings = []

    ckpt_dir = C.CHECKPOINT_DIR / "references" / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_refs):
        ref_seed = manifest["derived_seeds"]["reference_models"][i]
        rng = np.random.default_rng(ref_seed)
        pos = np.sort(rng.choice(n_pool, size=subset_size, replace=False))
        subset_positions[i] = pos
        in_out[i, pos] = 1

        model = MO.build_model(cfg["model"]["arch"], input_dim=input_dim, num_classes=meta["num_classes"])
        t0 = time.time()
        model, _ = T.train_classifier(
            model, pool_X[pos], pool_y[pos], cfg["model"], seed=ref_seed,
            epochs_override=epochs_override, log_prefix=f"[{dataset}/s{seed}/ref{i}] ",
        )
        timings.append(round(time.time() - t0, 1))
        torch.save(model.state_dict(), ckpt_dir / f"ref_{i}_seed{seed}.pt")

        ref_probs_pool[i] = T.forward_logits(model, pool_X).astype(np.float32)
        # softmax
        z = ref_probs_pool[i] - ref_probs_pool[i].max(axis=1, keepdims=True)
        p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
        ref_probs_pool[i] = p
        for arr, X in ((ref_probs_me[i], me_X), (ref_probs_nm[i], nm_X)):
            z = T.forward_logits(model, X).astype(np.float32)
            z = z - z.max(axis=1, keepdims=True)
            p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
            arr[:] = p

    bundle_dir = C.BUNDLE_DIR / dataset / f"seed{seed}"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        bundle_dir / "reference_bundle.npz",
        ref_probs_pool=ref_probs_pool,
        ref_probs_member_eval=ref_probs_me,
        ref_probs_nonmember=ref_probs_nm,
        in_out_matrix=in_out,
        subset_positions=subset_positions,
        pool_indices=pool_gids,
        population_pool_positions=pop_pos,
        population_indices=pop_gids,
        ref_train_seconds=np.array(timings, dtype=np.float64),
        git_commit=manifest["generated"]["git_commit"],
    )
    print(f"[reference-bundle] {dataset} seed{seed} saved: {num_refs} refs, pool {n_pool}, subset {subset_size}")
    return ref_probs_pool


def load_reference_bundle(dataset: str, seed: int) -> Dict[str, np.ndarray]:
    return dict(np.load(C.BUNDLE_DIR / dataset / f"seed{seed}" / "reference_bundle.npz"))
