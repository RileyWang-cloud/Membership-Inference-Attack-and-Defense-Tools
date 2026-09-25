"""W4b：模型对全分区的输出缓存（方案 §7）。

cache/target_outputs/<dataset>/seed<k>/<defense>/<partition>_{logits,probs,loss,labels}.npy
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch.nn as nn

from .config import CACHE_DIR
from .training import forward_logits, per_sample_ce_loss

# §7 规定的缓存分区（target 侧一次算全）
CACHED_PARTITIONS = [
    "target_member_eval", "target_nonmember", "target_val",
    "shadow_train", "shadow_test", "auxiliary", "rmia_population",
]


def compute_outputs(model: nn.Module, X: np.ndarray, y: np.ndarray, device=None) -> Dict[str, np.ndarray]:
    logits = forward_logits(model, X, device=device).astype(np.float32)
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    return {
        "logits": logits,
        "probs": probs.astype(np.float64),
        "loss": per_sample_ce_loss(logits, y),
        "labels": y.astype(np.int64),
    }


def cache_partition_outputs(model: nn.Module, partitions: Dict[str, Any], defense: str,
                            dataset: str, seed: int, device=None,
                            partitions_to_cache=CACHED_PARTITIONS) -> Dict[str, Dict[str, np.ndarray]]:
    """计算并落盘 target（或 defended target）对指定分区的输出。"""
    out_dir = CACHE_DIR / dataset / f"seed{seed}" / defense
    out_dir.mkdir(parents=True, exist_ok=True)
    cached = {}
    for name in partitions_to_cache:
        if name not in partitions:
            continue
        X, y = partitions[name]
        outs = compute_outputs(model, X, y, device=device)
        for kind, arr in outs.items():
            np.save(out_dir / f"{name}_{kind}.npy", arr)
        cached[name] = outs
        print(f"[cache] {dataset}/seed{seed}/{defense}/{name}: {len(y)} rows")
    return cached


def load_cached_outputs(dataset: str, seed: int, defense: str,
                        partitions=CACHED_PARTITIONS) -> Dict[str, Dict[str, np.ndarray]]:
    out_dir = CACHE_DIR / dataset / f"seed{seed}" / defense
    cached = {}
    for name in partitions:
        d = {}
        for kind in ("logits", "probs", "loss", "labels"):
            p = out_dir / f"{name}_{kind}.npy"
            if p.exists():
                d[kind] = np.load(p)
        if d:
            cached[name] = d
    return cached
