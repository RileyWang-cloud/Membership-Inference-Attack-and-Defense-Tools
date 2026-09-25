"""Shared plumbing for the 4090-side CIFAR-10 MIA benchmark (plan v2 §16).

Protocol constants (split sizes, seeds, training recipe) come from the
experiment plan and must not be changed per-run; they are duplicated in
configs/cifar10_resnet18.yaml which is the single runtime config source.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmark"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DATASET = "cifar10"
NUM_CLASSES = 10
NUM_WORKERS = 2          # lab-vm-71 has 8 cores / 7.8G RAM; keep loaders light
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ---------------------------------------------------------------- partitions
IMAGE_SPLITS: Dict[str, int] = {
    "target_train": 20000,
    "target_val": 5000,
    "shadow_train": 10000,
    "shadow_test": 10000,
    "auxiliary": 5000,
}
MEMBER_EVAL_SIZE = 10000     # subset of target_train, class-stratified
POPULATION_SIZE = 2500       # RMIA population, subset of reference_pool
REFERENCE_POOL_PARTS = ["shadow_train", "shadow_test", "auxiliary"]  # 25,000
NUM_REFS = 4
REF_SUBSET_RATIO = 0.5

EVAL_PARTITIONS = [
    "target_member_eval",
    "target_nonmember",
    "target_val",
    "shadow_train",
    "shadow_test",
    "auxiliary",
    "rmia_population",
]


def derive_seeds(seed: int) -> Dict[str, Any]:
    """Plan §3: shadow = 100*seed+50, ref_i = 100*seed+i (i = 0..3)."""
    return {
        "seed": seed,
        "shadow_seed": 100 * seed + 50,
        "reference_seeds": [100 * seed + i for i in range(NUM_REFS)],
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_of_array(array: np.ndarray) -> str:
    hasher = hashlib.sha256()
    flat = np.ascontiguousarray(array, dtype=np.float32).ravel()
    step = 1 << 24  # hash in 16MB chunks
    for start in range(0, flat.nbytes, step):
        hasher.update(flat.view(np.uint8)[start : start + step].tobytes())
    return hasher.hexdigest()


# ------------------------------------------------------------------- paths
def splits_dir() -> Path:
    return BENCH_ROOT / "splits" / DATASET


def manifest_path(seed: int) -> Path:
    return splits_dir() / f"seed{seed}" / "manifest.json"


def load_manifest(seed: int) -> Dict[str, Any]:
    path = manifest_path(seed)
    if not path.exists():
        raise FileNotFoundError(f"manifest not found: {path} (run build_manifest.py first)")
    with open(path) as handle:
        return json.load(handle)


def cache_dir(seed: int, defense: str = "clean") -> Path:
    return BENCH_ROOT / "cache" / "target_outputs" / DATASET / f"seed{seed}" / defense


def bundle_dir(seed: int) -> Path:
    return BENCH_ROOT / "bundles" / DATASET / f"seed{seed}"


def checkpoint_dir(kind: str) -> Path:
    path = BENCH_ROOT / "checkpoints" / kind
    # callers torch.save straight into this dir; create it so a fresh
    # workspace (or an archived checkpoints/ tree) cannot crash the save
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(kind: str) -> Path:
    return BENCH_ROOT / "results" / kind


def logs_dir() -> Path:
    return BENCH_ROOT / "logs"


class Timer:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self) -> "Timer":
        self.start = time.time()
        print(f"[timer] {self.label} ...", flush=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        seconds = time.time() - self.start
        print(f"[timer] {self.label} done in {seconds:.1f}s ({seconds / 60:.1f} min)", flush=True)


def log_to_file(message: str, name: str = "pipeline") -> None:
    logs_dir().mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(logs_dir() / f"{name}.log", "a") as handle:
        handle.write(f"[{stamp}] {message}\n")


def env_versions() -> Dict[str, str]:
    import torchvision  # local import to avoid cost when unused

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda": torch.version.cuda or "cpu",
        "cudnn": str(torch.backends.cudnn.version()) if torch.cuda.is_available() else "n/a",
    }


def git_commit() -> str:
    """Current commit hash (deliverables spec: recorded into every result row)."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Any:
    with open(path) as handle:
        return json.load(handle)


def finish_marker(kind: str, name: str) -> Path:
    return BENCH_ROOT / "logs" / "done" / f"{kind}_{name}.done"


def mark_done(kind: str, name: str) -> None:
    marker = finish_marker(kind, name)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))


def is_done(kind: str, name: str) -> bool:
    return finish_marker(kind, name).exists()


# ------------------------------------------------------- global index space
# Global index convention (manifest): index i in [0, 50000) -> CIFAR-10 train
# set position i; index 50000 + j -> CIFAR-10 official test set position j.
TRAIN_SIZE = 50000
TEST_SIZE = 10000


def global_to_position(index: int) -> tuple[str, int]:
    index = int(index)
    if index < TRAIN_SIZE:
        return "train", index
    if index < TRAIN_SIZE + TEST_SIZE:
        return "test", index - TRAIN_SIZE
    raise IndexError(f"global index {index} out of range")
