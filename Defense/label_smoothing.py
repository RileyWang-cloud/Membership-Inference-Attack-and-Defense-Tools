"""Label-smoothing defense for classification models (benchmark D08).

Reference: Szegedy et al., "Rethinking the Inception Architecture for
Computer Vision", CVPR 2016. MIA mitigation effect studied among others by
"Label Smoothing and Logit Squashing" style evaluations.

Training-time defense: the standard cross-entropy objective is replaced by
the smoothed objective

    L = -(1 - alpha) * log p_y - alpha / K * sum_k log p_k

with smoothing strength ``alpha`` (benchmark default 0.1, plan §10.5).
Everything else (optimizer family, epochs, batch size, lr, scheduler)
defaults to the dataset's clean-target recipe (benchmark plan §10.2 rule 1:
training-time defenses inherit the clean recipe; the smoothing term is the
only algorithm-required deviation).
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from Defense._classification import (
    build_loader,
    classifier_metrics,
    make_model,
    predict_logits,
    resolve_device,
)
from Defense.base import BaseDefense, DefenseInput, DefenseOutput


class LabelSmoothingDefense(BaseDefense):
    """Train with label smoothing on the same data / recipe as the target."""

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 0.1,
        alpha: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        milestones: tuple = (50, 75),
        gamma: float = 0.1,
        optimizer: str = "sgd",
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.milestones = tuple(int(m) for m in milestones)
        self.gamma = float(gamma)
        self.optimizer = str(optimizer).lower()
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None
        self._effective_config: Dict[str, Any] = {}

    def fit(self, defense_input: DefenseInput) -> "LabelSmoothingDefense":
        if defense_input.model_factory is None:
            raise ValueError("LabelSmoothingDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("LabelSmoothingDefense requires train_data and train_labels.")

        config = self._merge_config(defense_input.defense_config)
        model = make_model(defense_input.model_factory, self.device)
        loader = build_loader(
            defense_input.train_data,
            defense_input.train_labels,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        optimizer = self._make_optimizer(model, config)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=list(config["milestones"]), gamma=config["gamma"]
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=config["alpha"])

        self.training_history = []
        start = time.time()
        for epoch in range(config["epochs"]):
            model.train()
            loss_sum, seen, correct = 0.0, 0, 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.item()) * len(batch_y)
                correct += int((logits.argmax(1) == batch_y).sum())
                seen += len(batch_y)
            scheduler.step()
            self.training_history.append(
                {"epoch": float(epoch + 1), "train_loss": loss_sum / max(seen, 1),
                 "train_accuracy": correct / max(seen, 1)}
            )
        self.last_runtime_seconds = time.time() - start
        self._effective_config = config
        self.defended_model = model.eval()
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("LabelSmoothingDefense must be fitted before infer().")
        if defense_input.samples is None or defense_input.labels is None:
            raise ValueError("LabelSmoothingDefense.infer requires samples and labels.")
        logits = predict_logits(
            self.defended_model,
            defense_input.samples,
            batch_size=int(self._effective_config.get("batch_size", self.batch_size)),
            device=self.device,
        )
        return DefenseOutput(
            defended_model=self.defended_model,
            protected_outputs=logits,
            metadata={
                "defense_name": "label_smoothing",
                "alpha": self._effective_config.get("alpha", self.alpha),
                "runtime_seconds": self.last_runtime_seconds,
            },
        )

    def evaluate(self, defense_output: DefenseOutput, defense_input: DefenseInput):
        return classifier_metrics(
            defense_output.protected_outputs,
            defense_input.labels,
        )

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "optimizer": self.optimizer,
        }
        config.update(dict(overrides))
        config["batch_size"] = int(config["batch_size"])
        config["epochs"] = int(config["epochs"])
        config["learning_rate"] = float(config["learning_rate"])
        config["alpha"] = float(config["alpha"])
        config["momentum"] = float(config["momentum"])
        config["weight_decay"] = float(config["weight_decay"])
        config["milestones"] = tuple(int(m) for m in config["milestones"])
        config["gamma"] = float(config["gamma"])
        config["optimizer"] = str(config["optimizer"]).lower()
        if config["batch_size"] <= 0 or config["epochs"] <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if not 0.0 <= config["alpha"] < 1.0:
            raise ValueError("alpha must be in [0, 1).")
        return config

    def _make_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
        if config["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
            )
        if config["optimizer"] != "sgd":
            raise ValueError("optimizer must be either 'sgd' or 'adam'.")
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )


__all__ = ["LabelSmoothingDefense"]
