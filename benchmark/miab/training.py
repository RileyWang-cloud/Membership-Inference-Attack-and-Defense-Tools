"""W4：通用训练循环（fp32、不开 AMP，方案 §23.1 计时口径）。"""

from __future__ import annotations

import random
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.int64))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=0)


def build_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], momentum=cfg.get("momentum", 0.0),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
    raise ValueError(f"unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    sch = cfg.get("scheduler")
    if not sch:
        return None
    if sch["name"] == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=sch["milestones"], gamma=sch.get("gamma", 0.1)
        )
    raise ValueError(f"unknown scheduler: {sch['name']}")


def forward_logits(model: nn.Module, X: np.ndarray, batch_size: int = 512, device=None) -> np.ndarray:
    """推理期前向，返回 (N, C) logits（float32）。device 缺省跟随模型参数所在设备。"""
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device) if device is not None else model_device
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
            outs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outs) if outs else np.empty((0, model.classifier.out_features if hasattr(model, "classifier") else 10))


def per_sample_ce_loss(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    import torch.nn.functional as F
    import torch

    lt = torch.as_tensor(logits, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.long)
    return F.cross_entropy(lt, yt, reduction="none").numpy()


def train_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_cfg: Dict[str, Any],
    seed: int,
    device=None,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    epochs_override: Optional[int] = None,
    log_prefix: str = "",
) -> Tuple[nn.Module, List[Dict[str, Any]]]:
    """按 §5 配方训练分类器；返回 (model, history)。"""
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(seed)
    model = model.to(device)

    epochs = epochs_override if epochs_override is not None else int(train_cfg["epochs"])
    loader = make_loader(X_train, y_train, int(train_cfg["batch_size"]), shuffle=True, seed=seed)
    optimizer = build_optimizer(model, train_cfg)
    scheduler = build_scheduler(optimizer, train_cfg)
    loss_fn = nn.CrossEntropyLoss()

    history: List[Dict[str, Any]] = []
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        total_loss, correct, seen = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            seen += len(yb)
        if scheduler is not None:
            scheduler.step()
        rec: Dict[str, Any] = {
            "epoch": ep, "train_loss": total_loss / max(seen, 1), "train_acc": correct / max(seen, 1),
            "seconds": round(time.time() - t0, 2),
        }
        if X_val is not None and y_val is not None:
            vl = forward_logits(model, X_val, device=device)
            rec["val_acc"] = float((vl.argmax(1) == y_val).mean())
            rec["val_loss"] = float(per_sample_ce_loss(vl, y_val).mean())
        history.append(rec)
        print(f"{log_prefix}epoch {ep:3d} loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f}"
              + (f" val_acc {rec['val_acc']:.4f}" if "val_acc" in rec else "")
              + f" ({rec['seconds']}s)", flush=True)
    return model, history
