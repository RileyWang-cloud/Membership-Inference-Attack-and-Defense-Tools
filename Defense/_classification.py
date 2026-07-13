"""Shared PyTorch helpers for classification defenses.

The public defense API intentionally accepts ``Any`` so it can wrap different
model and data implementations.  The defenses in this directory use these
small helpers to keep their algorithm-specific code focused on the paper.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def resolve_device(device: Optional[str]) -> torch.device:
    return torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )


def extract_logits(model_output: Any) -> torch.Tensor:
    """Return logits from common PyTorch model output conventions."""
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, Mapping):
        logits = model_output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(model_output, (tuple, list)) and model_output:
        if isinstance(model_output[0], torch.Tensor):
            return model_output[0]
    raise TypeError(
        "Classifier must return a logits Tensor, a tuple/list whose first item "
        "is logits, or a mapping containing a 'logits' Tensor."
    )


def to_feature_tensor(value: Any) -> torch.Tensor:
    if value is None:
        raise ValueError("Expected feature data, got None.")
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def to_label_tensor(value: Any) -> torch.Tensor:
    if value is None:
        raise ValueError("Expected labels, got None.")
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return tensor.reshape(-1).to(torch.long)


def build_loader(
    samples: Any,
    labels: Optional[Any] = None,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    features = to_feature_tensor(samples)
    if labels is None:
        dataset = TensorDataset(features)
    else:
        targets = to_label_tensor(labels)
        if len(features) != len(targets):
            raise ValueError("Feature and label counts must match.")
        dataset = TensorDataset(features, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def make_model(model_factory: Any, device: torch.device) -> nn.Module:
    model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch.nn.Module.")
    return model.to(device)


@torch.no_grad()
def predict_logits(
    model: nn.Module,
    samples: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    loader = build_loader(samples, batch_size=batch_size, shuffle=False)
    was_training = model.training
    model.eval()
    outputs: List[torch.Tensor] = []
    for (batch_x,) in loader:
        logits = extract_logits(model(batch_x.to(device)))
        if logits.ndim != 2:
            raise ValueError("Classifier logits must have shape (batch, classes).")
        outputs.append(logits.detach().cpu())
    model.train(was_training)
    if not outputs:
        raise ValueError("Cannot predict an empty sample collection.")
    return torch.cat(outputs, dim=0)


def predict_labels(
    model: nn.Module,
    samples: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits = predict_logits(model, samples, device=device, batch_size=batch_size)
    return logits.argmax(dim=1).numpy().astype(np.int64)


@torch.no_grad()
def classifier_metrics(
    model: nn.Module,
    samples: Any,
    labels: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    logits = predict_logits(model, samples, device=device, batch_size=batch_size)
    targets = to_label_tensor(labels)
    if len(logits) != len(targets):
        raise ValueError("Prediction and label counts must match.")
    loss = F.cross_entropy(logits, targets, reduction="mean")
    accuracy = (logits.argmax(dim=1) == targets).float().mean()
    return {"loss": float(loss.item()), "accuracy": float(accuracy.item())}


@torch.no_grad()
def per_sample_losses(
    model: nn.Module,
    samples: Any,
    labels: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    logits = predict_logits(model, samples, device=device, batch_size=batch_size)
    targets = to_label_tensor(labels)
    if len(logits) != len(targets):
        raise ValueError("Prediction and label counts must match.")
    return F.cross_entropy(logits, targets, reduction="none").numpy()


def loss_mia_metrics(
    model: nn.Module,
    train_data: Any,
    train_labels: Any,
    test_data: Any,
    test_labels: Any,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    """Evaluate the standard per-sample-loss membership signal."""
    member_losses = per_sample_losses(
        model, train_data, train_labels, device=device, batch_size=batch_size
    )
    nonmember_losses = per_sample_losses(
        model, test_data, test_labels, device=device, batch_size=batch_size
    )
    membership = np.concatenate(
        [np.ones(len(member_losses), dtype=np.int64), np.zeros(len(nonmember_losses), dtype=np.int64)]
    )
    scores = -np.concatenate([member_losses, nonmember_losses])
    return {
        "loss_mia_auroc": binary_auroc(membership, scores),
        "mean_member_loss": float(member_losses.mean()),
        "mean_nonmember_loss": float(nonmember_losses.mean()),
        "loss_generalization_gap": float(nonmember_losses.mean() - member_losses.mean()),
    }


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC via average ranks, including correct handling of tied scores."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1).astype(np.float64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end

    rank_sum = float(ranks[positives].sum())
    return float((rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def probability_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum(dim=1)


__all__ = [
    "binary_auroc",
    "build_loader",
    "classifier_metrics",
    "extract_logits",
    "loss_mia_metrics",
    "make_model",
    "per_sample_losses",
    "predict_labels",
    "predict_logits",
    "probability_entropy",
    "resolve_device",
    "to_feature_tensor",
    "to_label_tensor",
]
