"""Clean-target training loop + full-partition output caching (plan §7, W4).

Recipe (plan §5.3, identical for target / shadow / reference models):
    SGD lr=0.1 momentum=0.9 weight_decay=5e-4 batch=128 epochs=100
    MultiStepLR milestones=[50, 75] gamma=0.1, no augmentation, no AMP
    (fp32 timing basis, plan §23.1). Clean targets use no privacy
    regularization of any kind (plan §5.4).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from benchmark.bmk import common
from benchmark.bmk.data import gather_global, load_cifar10_tensors


class TrainingDivergedError(RuntimeError):
    """Training hit NaN/inf loss — the model is dead and must be retrained."""

RECIPE: Dict[str, object] = {
    "optimizer": "sgd",
    "lr": 0.1,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "batch_size": 128,
    "epochs": 100,
    "scheduler": {"name": "MultiStepLR", "milestones": [50, 75], "gamma": 0.1},
    "augmentation": "none",
    "precision": "fp32",
}


def make_loader(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def evaluate_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device: torch.device,
                      batch_size: int = 512) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            logits = model(X[start : start + batch_size].to(device))
            correct += int((logits.argmax(1).cpu() == y[start : start + batch_size]).sum())
    return correct / len(X)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    seed: int,
    device: torch.device,
    val_X: Optional[torch.Tensor] = None,
    val_y: Optional[torch.Tensor] = None,
    epochs: int = 100,
    log_name: str = "train",
) -> Dict[str, object]:
    """Train with the fixed §5.3 recipe; returns history dict."""
    common.set_global_seed(seed)
    model = model.to(device)
    loader = make_loader(X, y, 128, shuffle=True)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75], gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss, seen = 0.0, 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device, non_blocking=False), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * len(batch_y)
            seen += len(batch_y)
        scheduler.step()
        epoch_loss = running_loss / seen
        if not math.isfinite(epoch_loss):
            raise TrainingDivergedError(
                f"{log_name}: loss became non-finite at epoch {epoch} "
                "(fp32 SGD lr=0.1 no-warmup divergence)"
            )
        train_acc = evaluate_accuracy(model, X, y, device)
        val_acc = (
            evaluate_accuracy(model, val_X, val_y, device)
            if val_X is not None
            else float("nan")
        )
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        print(
            f"[{log_name}] epoch {epoch:3d}/{epochs} loss={running_loss / seen:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
    model.eval()
    return history


def train_model_with_retry(
    make_model: Callable[[], nn.Module],
    *,
    seed: int,
    max_tries: int = 3,
    **train_kwargs,
) -> Tuple[nn.Module, Dict[str, object], int, List[dict]]:
    """train_model + auto-retry on fp32 divergence (hyperparameters untouched).

    The fixed §5.3 recipe (SGD lr=0.1, no warmup, no clipping) occasionally
    diverges to NaN depending on the RNG draw.  Attempt 0 keeps the nominal
    seed (cudnn's non-deterministic kernels make a rerun a fresh draw);
    attempts 1+ reseed with seed+1000*attempt, which is recorded in the
    checkpoint/bundle stats so the deliverable states the actual seed used.
    """
    attempts: List[dict] = []
    for attempt in range(max_tries):
        attempt_seed = seed + 1000 * attempt
        model = make_model()
        try:
            history = train_model(model, seed=attempt_seed, **train_kwargs)
            return model, history, attempt_seed, attempts
        except TrainingDivergedError as exc:
            attempts.append({"attempt": attempt, "seed": attempt_seed, "reason": str(exc)})
            print(f"[diverged] {exc} -> retry (attempt {attempt + 1}/{max_tries})")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError(f"training diverged {max_tries} times: {attempts}")


def partition_indices(manifest: Dict[str, object], name: str) -> np.ndarray:
    """Resolve a partition name to global indices (partitions dict or the
    top-level rmia_population entry)."""
    if name in manifest["partitions"]:
        return np.asarray(manifest["partitions"][name]["global_indices"], dtype=np.int64)
    if name == "rmia_population":
        return np.asarray(manifest["rmia_population"]["global_indices"], dtype=np.int64)
    raise KeyError(f"unknown partition {name}")


@torch.no_grad()
def cache_partition_outputs(
    model: nn.Module,
    seed: int,
    defense: str,
    device: torch.device,
    tensors=None,
    batch_size: int = 512,
) -> Dict[str, float]:
    """Cache logits/probs/loss/labels for every §7 partition (plan STEP 3)."""
    manifest = common.load_manifest(seed)
    out_dir = common.cache_dir(seed, defense)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tensors is None:
        tensors = load_cifar10_tensors()
    model = model.to(device)
    model.eval()
    summary: Dict[str, float] = {}
    for partition in common.EVAL_PARTITIONS:
        X, y = gather_global(partition_indices(manifest, partition), tensors)
        logits_list, losses_list = [], []
        for start in range(0, len(X), batch_size):
            batch_logits = model(X[start : start + batch_size].to(device))
            batch_logits = batch_logits.float().cpu()
            batch_losses = nn.functional.cross_entropy(batch_logits, y[start : start + batch_size], reduction="none")
            logits_list.append(batch_logits)
            losses_list.append(batch_losses)
        logits = torch.cat(logits_list)
        losses = torch.cat(losses_list)
        probs = torch.softmax(logits, dim=1)
        np.save(out_dir / f"{partition}_logits.npy", logits.numpy())
        np.save(out_dir / f"{partition}_probs.npy", probs.numpy())
        np.save(out_dir / f"{partition}_loss.npy", losses.numpy())
        np.save(out_dir / f"{partition}_labels.npy", y.numpy())
        summary[f"{partition}_acc"] = float((logits.argmax(1) == y).float().mean())
    # test-partition accuracy doubles as utility metric (target_nonmember = official test)
    summary["test_acc"] = summary["target_nonmember_acc"]
    common.write_json(out_dir / "summary.json", summary)
    return summary


def load_cache(seed: int, defense: str, partition: str) -> Dict[str, np.ndarray]:
    out_dir = common.cache_dir(seed, defense)
    return {
        "logits": np.load(out_dir / f"{partition}_logits.npy"),
        "probs": np.load(out_dir / f"{partition}_probs.npy"),
        "loss": np.load(out_dir / f"{partition}_loss.npy"),
        "labels": np.load(out_dir / f"{partition}_labels.npy"),
    }
