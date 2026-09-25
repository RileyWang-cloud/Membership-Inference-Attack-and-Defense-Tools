"""W2：数据加载器。

- Purchase / Texas：官方 tgz 解析（协议对齐 Shokri et al. 遗留代码的格式：
  Purchase 每行 = label,f1..f600；Texas = texas/100/feats(6169 维) + labels），
  以固定 DATA_SEED 从全量中采样 universe（替代遗留代码缺失的外部置换文件），
  解析结果缓存为 npz。
- MNIST / CIFAR-10：torchvision，归一化后以 float32 全量驻留内存。

全局索引（gid）约定（manifest 的唯一索引空间）：
- 表格：gid = universe 行号，范围 [0, N)
- 图像：gid ∈ [0, N_train) → 训练集行；gid ∈ [N_train, N_train+N_test) → 测试集行
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Tuple

import numpy as np

from .config import DATA_PROC_DIR, DATA_RAW_DIR, DATASET_META, TV_ROOT

# universe 构造种子：与 benchmark seed 无关的固定常数，保证所有 seed 用同一 universe
DATA_SEED = 20260922
UNIVERSE_SIZES = {"purchase": 19720, "texas": 10669}

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def _sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_purchase() -> Dict[str, Any]:
    import pandas as pd

    raw = DATA_RAW_DIR / "dataset_purchase"
    df = pd.read_csv(raw, header=None, dtype=np.float32)
    arr = df.to_numpy(dtype=np.float32)
    y = arr[:, 0].astype(np.int64) - 1
    X = arr[:, 1:].astype(np.float32)
    return X, y, {"raw_file": str(raw), "raw_sha256": _sha256_file(raw), "raw_rows": int(arr.shape[0])}


def _build_texas() -> Dict[str, Any]:
    import pandas as pd

    feats_path = DATA_RAW_DIR / "texas" / "100" / "feats"
    labels_path = DATA_RAW_DIR / "texas" / "100" / "labels"
    X = pd.read_csv(feats_path, header=None, dtype=np.float32).to_numpy(dtype=np.float32)
    y = pd.read_csv(labels_path, header=None, dtype=np.int64).to_numpy(dtype=np.int64).reshape(-1) - 1
    prov = {
        "raw_file": str(feats_path),
        "raw_sha256": _sha256_file(feats_path),
        "labels_sha256": _sha256_file(labels_path),
        "raw_rows": int(X.shape[0]),
    }
    return X, y, prov


def ensure_tabular(dataset: str) -> Dict[str, Any]:
    """解析原始表格数据 → 固定 universe 采样 → 缓存 npz。返回加载结果。"""
    npz = DATA_PROC_DIR / f"{dataset}_universe.npz"
    meta_path = DATA_PROC_DIR / f"{dataset}_universe.json"
    if npz.exists() and meta_path.exists():
        data = np.load(npz)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return {"X": data["X"], "y": data["y"], "meta": meta}

    if dataset == "purchase":
        X_full, y_full, prov = _build_purchase()
    elif dataset == "texas":
        X_full, y_full, prov = _build_texas()
    else:
        raise ValueError(f"unknown tabular dataset: {dataset}")

    n_universe = UNIVERSE_SIZES[dataset]
    rng = np.random.default_rng(DATA_SEED)
    source_rows = np.sort(rng.choice(X_full.shape[0], size=n_universe, replace=False))
    X = np.ascontiguousarray(X_full[source_rows])
    y = np.ascontiguousarray(y_full[source_rows])

    meta = {
        "dataset": dataset,
        "data_seed": DATA_SEED,
        "universe_size": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "num_classes": int(y.max() + 1),
        "source_row_indices_sha": hashlib.sha256(source_rows.tobytes()).hexdigest(),
        "provenance": prov,
    }
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(npz, X=X, y=y, source_row_indices=source_rows)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return {"X": X, "y": y, "meta": meta}


def _load_torchvision(dataset: str):
    import torchvision
    from torchvision import datasets as tvd

    if dataset == "mnist":
        tr = tvd.MNIST(TV_ROOT, train=True, download=True)
        te = tvd.MNIST(TV_ROOT, train=False, download=True)
        Xtr = tr.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
        Xte = te.data.numpy().astype(np.float32).reshape(-1, 1, 28, 28) / 255.0
        Xtr = (Xtr - MNIST_MEAN) / MNIST_STD
        Xte = (Xte - MNIST_MEAN) / MNIST_STD
        ytr, yte = tr.targets.numpy().astype(np.int64), te.targets.numpy().astype(np.int64)
    elif dataset == "cifar10":
        tr = tvd.CIFAR10(TV_ROOT, train=True, download=True)
        te = tvd.CIFAR10(TV_ROOT, train=False, download=True)
        Xtr = tr.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0  # (N,3,32,32)
        Xte = te.data.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(CIFAR_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        Xtr = (Xtr - mean) / std
        Xte = (Xte - mean) / std
        ytr = np.asarray(tr.targets, dtype=np.int64)
        yte = np.asarray(te.targets, dtype=np.int64)
    else:
        raise ValueError(f"unknown image dataset: {dataset}")

    prov = {
        "source": "torchvision",
        "torchvision_version": torchvision.__version__,
        "train_rows": int(Xtr.shape[0]),
        "test_rows": int(Xte.shape[0]),
    }
    return Xtr, ytr, Xte, yte, prov


def load_dataset(dataset: str) -> Dict[str, Any]:
    """统一加载入口。返回:
    tabular: {X, y, meta}
    image:   {X_train, y_train, X_test, y_test, meta}
    并附 gid 解析所需信息（n_train）。
    """
    meta_info = DATASET_META[dataset]
    if meta_info["modality"] == "tabular":
        out = ensure_tabular(dataset)
        out["n_train"] = out["X"].shape[0]
        out["is_image"] = False
        return out

    Xtr, ytr, Xte, yte, prov = _load_torchvision(dataset)
    import hashlib as _h

    prov["train_sha256"] = _h.sha256(np.ascontiguousarray(Xtr).tobytes()).hexdigest()
    prov["test_sha256"] = _h.sha256(np.ascontiguousarray(Xte).tobytes()).hexdigest()
    return {
        "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
        "meta": {"dataset": dataset, "provenance": prov},
        "n_train": int(Xtr.shape[0]),
        "is_image": True,
    }


def resolve_gid(data: Dict[str, Any], gid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """把 manifest gid 翻译回 (X, y)。"""
    gid = np.asarray(gid, dtype=np.int64)
    if data["is_image"]:
        n_train = data["n_train"]
        # 保持 gid 顺序：train 侧 gid 原样、test 侧 gid 减去 n_train 后取测试集行
        X = np.empty(tuple([len(gid)]) + data["X_train"].shape[1:], dtype=np.float32)
        y = np.empty(len(gid), dtype=np.int64)
        m_tr, m_te = gid < n_train, gid >= n_train
        X[m_tr] = data["X_train"][gid[m_tr]]
        X[m_te] = data["X_test"][gid[m_te] - n_train]
        y[m_tr] = data["y_train"][gid[m_tr]]
        y[m_te] = data["y_test"][gid[m_te] - n_train]
        return X, y
    return data["X"][gid], data["y"][gid]


def partitions_from_manifest(data: Dict[str, Any], manifest: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """manifest["partitions"] 的 gid → (X, y) 字典。"""
    out = {}
    for name, part in manifest["partitions"].items():
        out[name] = resolve_gid(data, np.asarray(part["indices"], dtype=np.int64))
    for name in ("reference_pool", "rmia_population"):
        if name in manifest:
            out[name] = resolve_gid(data, np.asarray(manifest[name]["indices"], dtype=np.int64))
    return out
