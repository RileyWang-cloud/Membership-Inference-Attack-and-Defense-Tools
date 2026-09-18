"""
Training-time label smoothing defense for classification models.

Label smoothing trains the classifier against softly-distributed targets
((1 - smoothing) mass on the true class, the rest spread over the other
classes), flattening output confidence. Overconfident outputs on training
samples are the main signal metric-based membership inference attacks
exploit, so confidence-flattening regularizers are commonly evaluated as
MIA mitigations. The measured privacy effect is attack-model dependent
(e.g. "Be Careful What You Smooth For", ICLR 2024, shows label smoothing
can be a privacy shield or a catalyst), so benchmark this defense with the
platform's attacks instead of assuming a privacy gain.

This implementation follows the same self-contained style as dp_sgd.py:
- training-time defense
- PyTorch classification models
- standard mini-batch training, only the loss target is smoothed
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


class LabelSmoothingDefense(BaseDefense):
    """Training-time label smoothing regularization for classification models.

    defense_mode: training_time

    Required DefenseInput fields:
        - model_factory: callable returning a fresh torch.nn.Module
        - train_data / train_labels: classification training set

    Optional fields: val/test data for evaluation, samples/labels for
    protected predictions.

    Main output: DefenseOutput.defended_model (the label-smoothed trained
    model). Protected predictions for DefenseInput.samples are returned in
    DefenseOutput.protected_outputs.
    """

    name = "label_smoothing"
    defense_family = "regularization"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = ["model_factory", "train_data", "train_labels"]
    optional_input_keys = ["val_data", "val_labels", "test_data", "test_labels", "samples", "labels"]

    def __init__(
        self,
        smoothing: float = 0.1,
        batch_size: int = 128,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.smoothing = smoothing
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None

    def fit(self, defense_input: DefenseInput) -> "LabelSmoothingDefense":
        if defense_input.model_factory is None:
            raise ValueError("LabelSmoothingDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("LabelSmoothingDefense requires train_data and train_labels.")

        config = defense_input.defense_config
        smoothing = float(config.get("smoothing", self.smoothing))
        batch_size = int(config.get("batch_size", self.batch_size))
        epochs = int(config.get("epochs", self.epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))

        if not 0.0 <= smoothing < 1.0:
            raise ValueError(f"smoothing must be in [0.0, 1.0), got {smoothing}.")

        model = defense_input.model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module for LabelSmoothingDefense.")
        model = model.to(self.device)

        train_loader = _build_loader(
            defense_input.train_data,
            defense_input.train_labels,
            batch_size=batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing, reduction="none")

        self.training_history = []
        start_time = time.time()

        model.train()
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_example_count = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size_actual = batch_x.shape[0]

                optimizer.zero_grad()
                per_sample_losses = criterion(model(batch_x), batch_y)
                loss = per_sample_losses.mean()
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(per_sample_losses.detach().sum().cpu().item())
                epoch_example_count += batch_size_actual

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "train_loss": epoch_loss_sum / max(epoch_example_count, 1),
                }
            )

        self.last_runtime_seconds = time.time() - start_time
        self.defended_model = model.eval()
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("LabelSmoothingDefense must be fitted before infer().")

        protected_outputs = None
        if defense_input.samples is not None:
            protected_outputs = self._predict_labels(self.defended_model, defense_input.samples)

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.defended_model,
            protected_outputs=protected_outputs,
            artifacts={
                "training_history": self.training_history,
                "label_smoothing_config": {
                    "smoothing": self.smoothing,
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                },
            },
            intermediate_outputs={
                "training_history": self.training_history,
            },
            metadata={
                "defense_name": self.name,
                "defense_family": self.defense_family,
                "defense_mode": self.defense_mode,
            },
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        utility_metrics: Dict[str, float] = {}
        efficiency_metrics: Dict[str, float] = {}

        model = defense_output.defended_model
        if model is None:
            raise ValueError("No defended model available for evaluation.")

        if defense_input.test_data is not None and defense_input.test_labels is not None:
            test_preds = self._predict_labels(model, defense_input.test_data)
            test_labels = _to_numpy_1d(defense_input.test_labels)
            utility_metrics["test_accuracy"] = float(np.mean(test_preds == test_labels))

        if defense_input.train_data is not None and defense_input.train_labels is not None:
            train_preds = self._predict_labels(model, defense_input.train_data)
            train_labels = _to_numpy_1d(defense_input.train_labels)
            utility_metrics["train_accuracy"] = float(np.mean(train_preds == train_labels))

        if self.last_runtime_seconds is not None:
            efficiency_metrics["train_time"] = float(self.last_runtime_seconds)

        return DefenseEvaluationResult(
            utility_metrics=utility_metrics or None,
            privacy_metrics={
                "smoothing": float(defense_input.defense_config.get("smoothing", self.smoothing)),
            },
            efficiency_metrics=efficiency_metrics or None,
            extra_metrics=None,
        )

    def _predict_labels(self, model: nn.Module, samples: Any) -> np.ndarray:
        model.eval()
        loader = _build_predict_loader(samples, batch_size=self.batch_size)
        all_preds: List[torch.Tensor] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                logits = model(batch_x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.detach().cpu())

        return torch.cat(all_preds, dim=0).numpy().astype(np.int64)


def _build_loader(
    samples: Any,
    labels: Any,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    y_tensor = _to_tensor_1d(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _build_predict_loader(samples: Any, batch_size: int) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    dataset = TensorDataset(x_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


__all__ = [
    "LabelSmoothingDefense",
]
