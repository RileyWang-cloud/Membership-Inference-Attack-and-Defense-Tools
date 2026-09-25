"""W1：split manifest 生成与校验（方案 §4）。

规则：
- class-stratified 分配（largest remainder 逐类配额）
- 分区严格不重叠；target_member_eval ⊂ target_train
- reference_pool = shadow_train ∪ shadow_test ∪ auxiliary
- rmia_population ⊂ reference_pool，与攻击评估集不重叠
- 派生种子：shadow = 100*seed+50，ref_i = 100*seed+i（§3）
- 记录划分种子、数据 sha256、类别分布、git commit
"""

from __future__ import annotations

import datetime
import json
import subprocess
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np

from .config import manifest_path

BASE_PARTITIONS = ["target_train", "target_val", "shadow_train", "shadow_test", "auxiliary"]


def _git_commit() -> str:
    try:
        from .config import REPO_ROOT
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def stratified_alloc(y: np.ndarray, sizes: "OrderedDict[str, int]", rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """按类别分层把候选池切成给定尺寸的分区。

    每类配额 = largest remainder of n_c * S_p / N_pool（N_pool 为候选池总量，
    允许 total_alloc < N_pool，余量留给 reserve）。最后跨分区微调补齐目标尺寸。
    """
    pool_size = len(y)
    total_alloc = sum(sizes.values())
    assert total_alloc <= pool_size, "split sizes exceed pool"
    result: Dict[str, List[np.ndarray]] = {name: [] for name in sizes}

    # 1) 每类贡献总额 total_c（largest remainder 使 Σ total_c == total_alloc）
    classes = np.unique(y)
    n_cs = {int(c): int(np.sum(y == c)) for c in classes}
    exact_total = {int(c): n_cs[int(c)] * total_alloc / pool_size for c in classes}
    total_c = {int(c): int(np.floor(exact_total[int(c)])) for c in classes}
    remaining = total_alloc - sum(total_c.values())
    by_frac = sorted(classes, key=lambda c: exact_total[int(c)] - total_c[int(c)], reverse=True)
    for i in range(remaining):
        total_c[int(by_frac[i % len(by_frac)])] += 1
    for c in classes:
        total_c[int(c)] = min(total_c[int(c)], n_cs[int(c)])

    # 2) 类内把 total_c 按 largest remainder 分到各分区
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        n_c = len(idx_c)
        exact = {name: total_c[int(c)] * s / total_alloc for name, s in sizes.items()}
        quotas = {name: int(np.floor(exact[name])) for name in sizes}
        rem = total_c[int(c)] - sum(quotas.values())
        fracs = sorted(sizes.keys(), key=lambda name: exact[name] - quotas[name], reverse=True)
        i = 0
        while rem > 0:
            quotas[fracs[i % len(fracs)]] += 1
            rem -= 1
            i += 1
        start = 0
        for name, q in quotas.items():
            q = min(q, n_c - start)
            result[name].append(idx_c[start:start + q])
            start += q

    out = {name: (np.concatenate(v) if v else np.array([], dtype=np.int64)) for name, v in result.items()}
    # 逐分区对齐目标尺寸：先收超领样本，再补给欠额分区（打破的分层性仅为个位数样本）
    over: List[int] = []
    for name, target in sizes.items():
        arr = out[name]
        if len(arr) > target:
            rng.shuffle(arr)
            over.extend(arr[target:].tolist())
            out[name] = np.sort(arr[:target])
    for name, target in sizes.items():
        arr = out[name]
        if len(arr) < target:
            need = target - len(arr)
            take = np.array(over[:need], dtype=np.int64)
            over = over[need:]
            out[name] = np.sort(np.concatenate([arr, take])) if len(take) else arr
    assert not over, "allocation leftover — sizes exceed pool"
    return out


def stratified_subset(y_pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    """从候选池按类分层抽取固定大小子集（如 member_eval ⊂ target_train）。

    返回子集在候选池内的位置（已排序，保证确定性读取）。
    """
    classes = np.unique(y_pool)
    n = len(y_pool)
    base = {int(c): int(len(np.where(y_pool == c)[0]) * size // n) for c in classes}
    remaining = size - sum(base.values())
    order = sorted(
        classes,
        key=lambda c: len(np.where(y_pool == c)[0]) * size / n - base[int(c)],
        reverse=True,
    )
    for i in range(remaining):
        base[int(order[i % len(order)])] += 1
    picked: List[np.ndarray] = []
    used_all: List[np.ndarray] = []
    for c in classes:
        idx_c = np.where(y_pool == c)[0]
        rng.shuffle(idx_c)
        q = min(base[int(c)], len(idx_c))
        picked.append(idx_c[:q])
        used_all.append(idx_c)
    sel = np.concatenate(picked) if picked else np.array([], dtype=np.int64)
    if len(sel) < size:  # 极端类不平衡时的兜底：从剩余样本补齐
        rest = np.setdiff1d(np.concatenate(used_all), sel)
        rng.shuffle(rest)
        sel = np.concatenate([sel, rest[: size - len(sel)]])
    return np.sort(sel)


def _class_counts(y: np.ndarray, num_classes: int) -> Dict[str, int]:
    counts = np.bincount(y, minlength=num_classes)
    return {str(i): int(c) for i, c in enumerate(counts) if c > 0}


def build_manifest(dataset: str, seed: int, cfg: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    splits_cfg = cfg["data"]["splits"]
    num_classes = int(cfg["data"]["num_classes"])
    is_image = data["is_image"]

    partitions: Dict[str, Dict[str, Any]] = {}

    if is_image:
        n_train = data["n_train"]
        y_tr, y_te = data["y_train"], data["y_test"]
        base_sizes = OrderedDict((name, int(splits_cfg[name])) for name in BASE_PARTITIONS)
        alloc = stratified_alloc(y_tr, base_sizes, rng)
        used = np.concatenate(list(alloc.values()))
        reserve_idx = np.setdiff1d(np.arange(n_train), used)

        # member_eval ⊂ target_train
        tt_local = alloc["target_train"]
        me_local = stratified_subset(y_tr[tt_local], int(splits_cfg["target_member_eval"]), rng)

        # nonmember：官方测试集内分层抽取
        nm_local = stratified_subset(y_te, int(splits_cfg["target_nonmember"]), rng)

        gid_map = {
            "target_train": (alloc["target_train"], alloc["target_train"], "train"),
            "target_val": (alloc["target_val"], alloc["target_val"], "train"),
            "shadow_train": (alloc["shadow_train"], alloc["shadow_train"], "train"),
            "shadow_test": (alloc["shadow_test"], alloc["shadow_test"], "train"),
            "auxiliary": (alloc["auxiliary"], alloc["auxiliary"], "train"),
            "target_member_eval": (tt_local[me_local], tt_local[me_local], "train"),
            "target_nonmember": (nm_local + n_train, nm_local, "test"),
        }
        reserve = {"indices": reserve_idx.tolist(), "source": "train"}
        y_lookup = {"train": y_tr, "test": y_te}
    else:
        X_size = data["X"].shape[0]
        y_all = data["y"]
        base_sizes = OrderedDict((name, int(splits_cfg[name])) for name in BASE_PARTITIONS)
        base_sizes["target_nonmember"] = int(splits_cfg["target_nonmember"])
        alloc = stratified_alloc(y_all, base_sizes, rng)
        used = np.concatenate(list(alloc.values()))
        reserve_idx = np.setdiff1d(np.arange(X_size), used)

        tt_local = alloc["target_train"]
        me_local = stratified_subset(y_all[tt_local], int(splits_cfg["target_member_eval"]), rng)

        gid_map = {
            "target_train": (alloc["target_train"], alloc["target_train"], "universe"),
            "target_val": (alloc["target_val"], alloc["target_val"], "universe"),
            "shadow_train": (alloc["shadow_train"], alloc["shadow_train"], "universe"),
            "shadow_test": (alloc["shadow_test"], alloc["shadow_test"], "universe"),
            "auxiliary": (alloc["auxiliary"], alloc["auxiliary"], "universe"),
            "target_member_eval": (tt_local[me_local], tt_local[me_local], "universe"),
            "target_nonmember": (alloc["target_nonmember"], alloc["target_nonmember"], "universe"),
        }
        reserve = {"indices": reserve_idx.tolist(), "source": "universe"}
        y_lookup = {"universe": y_all}

    for name, (gids, local, src) in gid_map.items():
        partitions[name] = {
            "indices": np.sort(np.asarray(gids, dtype=np.int64)).tolist(),
            "size": int(len(gids)),
            "source": src,
            "class_counts": _class_counts(y_lookup[src][np.asarray(local, dtype=np.int64)], num_classes),
        }

    # reference_pool 与 rmia_population（pool 排序保证 searchsorted 定位有效）
    pool = np.sort(np.concatenate(
        [np.asarray(partitions[p]["indices"], dtype=np.int64) for p in ("shadow_train", "shadow_test", "auxiliary")]
    ))
    if is_image:
        pool_X, pool_y = None, None
        # 用 gid 解析拿 y（避免在这里重复实现）
        from .data import resolve_gid
        _, pool_y = resolve_gid(data, pool)
    else:
        pool_y = data["y"][pool]
    pop_size = int(cfg["population"]["size"])
    pop_pool_pos = stratified_subset(pool_y, pop_size, rng)  # pool 内位置
    population = pool[np.sort(pop_pool_pos)]

    manifest = {
        "schema_version": 1,
        "dataset": dataset,
        "seed": int(seed),
        "derived_seeds": {
            "split": int(seed),
            "shadow_model": 100 * int(seed) + 50,
            "reference_models": [100 * int(seed) + i for i in range(int(cfg["reference"]["num_models"]))],
        },
        "dataset_meta": data["meta"],
        "partitions": partitions,
        "reference_pool": {
            "indices": pool.tolist(),
            "size": int(len(pool)),
            "from": ["shadow_train", "shadow_test", "auxiliary"],
            "subset_ratio": float(cfg["reference"]["subset_ratio"]),
        },
        "rmia_population": {
            "indices": population.tolist(),
            "size": int(len(population)),
            "from": "reference_pool",
        },
        "reserve": reserve,
        "attack_eval": {
            "member_partition": "target_member_eval",
            "nonmember_partition": "target_nonmember",
        },
        "generated": {
            "time": datetime.datetime.now().isoformat(timespec="seconds"),
            "git_commit": _git_commit(),
        },
    }
    validate_manifest(manifest)
    return manifest


def validate_manifest(m: Dict[str, Any]) -> None:
    parts = {k: np.asarray(v["indices"], dtype=np.int64) for k, v in m["partitions"].items()}
    base = [parts[p] for p in BASE_PARTITIONS]
    nm = parts["target_nonmember"]

    # 基本分区互斥
    all_base = np.concatenate(base + [nm])
    assert len(np.unique(all_base)) == len(all_base), "partitions overlap"

    # member_eval ⊂ target_train
    me, tt = parts["target_member_eval"], parts["target_train"]
    assert len(np.intersect1d(me, tt)) == len(me), "member_eval not subset of target_train"

    # pool 恰为三分区并集
    pool = np.asarray(m["reference_pool"]["indices"], dtype=np.int64)
    union = np.concatenate([parts["shadow_train"], parts["shadow_test"], parts["auxiliary"]])
    assert len(np.intersect1d(pool, parts["target_train"])) == 0, "pool overlaps target_train"
    assert len(pool) == len(union), "pool size mismatch"

    # population ⊂ pool 且与评估集不重叠
    pop = np.asarray(m["rmia_population"]["indices"], dtype=np.int64)
    assert len(np.intersect1d(pop, pool)) == len(pop), "population not subset of pool"
    assert len(np.intersect1d(pop, me)) == 0 and len(np.intersect1d(pop, nm)) == 0

    # 尺寸核对
    for name, p in m["partitions"].items():
        assert p["size"] == len(p["indices"]), f"size mismatch: {name}"


def save_manifest(m: Dict[str, Any]) -> None:
    path = manifest_path(m["dataset"], m["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    print(f"[manifest] written: {path}")
