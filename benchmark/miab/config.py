"""路径与配置加载。目录约定严格对齐方案 v2 §16。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_ROOT = REPO_ROOT / "benchmark"

CONFIG_DIR = BENCH_ROOT / "configs"
SPLITS_DIR = BENCH_ROOT / "splits"
CHECKPOINT_DIR = BENCH_ROOT / "checkpoints"
BUNDLE_DIR = BENCH_ROOT / "bundles"
CACHE_DIR = BENCH_ROOT / "cache"
RESULTS_DIR = BENCH_ROOT / "results"
LOGS_DIR = BENCH_ROOT / "logs"
DATA_RAW_DIR = BENCH_ROOT / "data" / "raw"
DATA_PROC_DIR = BENCH_ROOT / "data" / "processed"
TV_ROOT = BENCH_ROOT / "data" / "torchvision"

# 数据集元信息（W2/W3 依赖；num_classes 与输入规格按方案 §2/§5）
DATASET_META: Dict[str, Dict[str, Any]] = {
    "purchase": {"modality": "tabular", "num_classes": 100, "input_dim": 600, "is_image": False},
    "texas": {"modality": "tabular", "num_classes": 100, "input_dim": 6169, "is_image": False},
    "mnist": {"modality": "image", "num_classes": 10, "shape": (1, 28, 28), "is_image": True},
    "cifar10": {"modality": "image", "num_classes": 10, "shape": (3, 32, 32), "is_image": True},
}

CONFIG_NAMES = {
    "purchase": "purchase_mlp",
    "texas": "texas_mlp",
    "mnist": "mnist_cnn",
    "cifar10": "cifar10_resnet18",
}


def load_config(dataset: str) -> Dict[str, Any]:
    name = CONFIG_NAMES[dataset]
    path = CONFIG_DIR / f"{name}.yaml"
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_name"] = name
    cfg["_path"] = str(path)
    return cfg


def manifest_path(dataset: str, seed: int) -> Path:
    return SPLITS_DIR / dataset / f"seed{seed}" / "manifest.json"


def load_manifest(dataset: str, seed: int) -> Dict[str, Any]:
    import json

    with open(manifest_path(dataset, seed), "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dirs() -> None:
    for d in (
        SPLITS_DIR, CHECKPOINT_DIR, BUNDLE_DIR, CACHE_DIR, RESULTS_DIR,
        LOGS_DIR, DATA_PROC_DIR, TV_ROOT,
    ):
        d.mkdir(parents=True, exist_ok=True)
