"""
Training-time mixup defense for classification models (benchmark 方案 §10.5).

Batch 内凸组合 ``x~ = λ·x_i + (1−λ)·x_j``，``λ ~ Beta(α, α)``，
损失 = λ·CE(f(x~), y_i) + (1−λ)·CE(f(x~), y_j)（软标签形式）。
适用于表格与图像分类，其余训练配置沿用 Clean Target 配方（§10.2）。

接口与 label_smoothing.py / dp_sgd.py 同风格（BaseDefense 子类）。
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Defense.base import BaseDefense, DefenseInput, DefenseOutput


def _build_loader(X: Any, y: Any, batch_size: int, shuffle: bool) -> DataLoader:
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y), dtype=torch.long)
    return DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=shuffle)


class MixupDefense(BaseDefense):
    """Training-time mixup regularization.

    defense_mode: training_time

    Required DefenseInput fields:
        - model_factory: callable returning a fresh torch.nn.Module
        - train_data / train_labels

    defense_config overrides: alpha, batch_size, epochs, learning_rate, seed
    """

    name = "mixup"
    defense_family = "regularization"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = ["model_factory", "train_data", "train_labels"]
    optional_input_keys = ["val_data", "val_labels", "test_data", "test_labels", "samples", "labels"]

    def __init__(
        self,
        alpha: float = 1.0,
        batch_size: int = 128,
        epochs: int = 20,
        learning_rate: float = 1e-3,
        device: Optional[str] = None,
    ) -> None:
        self.alpha = alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None

    def fit(self, defense_input: DefenseInput) -> "MixupDefense":
        if defense_input.model_factory is None:
            raise ValueError("MixupDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("MixupDefense requires train_data and train_labels.")

        config = defense_input.defense_config or {}
        alpha = float(config.get("alpha", self.alpha))
        batch_size = int(config.get("batch_size", self.batch_size))
        epochs = int(config.get("epochs", self.epochs))
        learning_rate = float(config.get("learning_rate", self.learning_rate))
        if "seed" in config:
            torch.manual_seed(int(config["seed"]))

        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}.")

        model = defense_input.model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module for MixupDefense.")
        model = model.to(self.device)

        train_loader = _build_loader(defense_input.train_data, defense_input.train_labels,
                                     batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.training_history = []
        start_time = time.time()

        model.train()
        for epoch in range(epochs):
            epoch_loss_sum, epoch_count = 0.0, 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                b = batch_x.shape[0]

                lam = float(torch.distributions.Beta(alpha, alpha).sample())
                perm = torch.randperm(b, device=self.device)
                mixed_x = lam * batch_x + (1.0 - lam) * batch_x[perm]

                optimizer.zero_grad()
                logits = model(mixed_x)
                loss = lam * criterion(logits, batch_y) + (1.0 - lam) * criterion(logits, batch_y[perm])
                loss.backward()
                optimizer.step()

                epoch_loss_sum += float(loss.detach().cpu().item()) * b
                epoch_count += b

            self.training_history.append({
                "epoch": float(epoch + 1),
                "train_loss": epoch_loss_sum / max(epoch_count, 1),
            })

        self.last_runtime_seconds = time.time() - start_time
        self.defended_model = model.eval()
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("MixupDefense must be fitted before infer().")

        protected_outputs = None
        if defense_input.samples is not None:
            X_t = torch.as_tensor(np.asarray(defense_input.samples), dtype=torch.float32).to(self.device)
            with torch.no_grad():
                self.defended_model.eval()
                protected_outputs = self.defended_model(X_t).detach().cpu().numpy()

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.defended_model,
            protected_outputs=protected_outputs,
            transformed_data=None,
            artifacts={"alpha": self.alpha, "epochs": self.epochs},
            intermediate_outputs={"training_history": list(self.training_history)},
            metadata={"defense_name": self.name, "defense_mode": self.defense_mode},
        )
