"""RelaxLoss defense for PyTorch classifiers.

Reference: Chen, Yu, and Fritz, "RelaxLoss: Defending Membership Inference
Attacks without Losing Utility", ICLR 2022.

The implementation follows the authors' alternating objective: even epochs
move the mean cross-entropy toward a target level ``alpha``; odd epochs use
ordinary descent above that level and posterior flattening below it.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Defense._classification import (
    build_loader,
    classifier_metrics,
    extract_logits,
    loss_mia_metrics,
    make_model,
    predict_labels,
    resolve_device,
)
from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


def relax_loss_objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    epoch: int,
    alpha: float,
    upper: float,
) -> torch.Tensor:
    """Compute the RelaxLoss objective used by the official implementation."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("RelaxLoss requires logits with at least two classes.")

    per_sample_ce = F.cross_entropy(logits, targets, reduction="none")
    mean_ce = per_sample_ce.mean()
    if epoch % 2 == 0:
        return torch.abs(mean_ce - alpha)
    if float(mean_ce.detach().item()) > alpha:
        return mean_ce

    num_classes = logits.shape[1]
    probabilities = torch.softmax(logits, dim=1)
    target_confidence = probabilities.gather(1, targets[:, None]).squeeze(1)
    target_confidence = torch.clamp(target_confidence, min=0.0, max=upper)
    other_confidence = (1.0 - target_confidence) / float(num_classes - 1)
    one_hot = F.one_hot(targets, num_classes=num_classes).to(logits.dtype)
    soft_targets = (
        one_hot * target_confidence[:, None]
        + (1.0 - one_hot) * other_confidence[:, None]
    )

    soft_ce = -(soft_targets * F.log_softmax(logits, dim=1)).sum(dim=1)
    correct = (logits.argmax(dim=1) == targets).to(logits.dtype)
    return ((1.0 - correct) * soft_ce - per_sample_ce).mean()


class RelaxLossDefense(BaseDefense):
    """Training-time RelaxLoss defense for multi-class classifiers.

    Required ``DefenseInput`` fields: ``model_factory``, ``train_data``, and
    ``train_labels``.  The main output is ``defended_model``.
    """

    name = "relax_loss"
    defense_family = "loss_regularization"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = ["model_factory", "train_data", "train_labels"]
    optional_input_keys = ["test_data", "test_labels", "samples", "labels"]

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 0.1,
        alpha: float = 1.0,
        upper: float = 1.0,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        milestones: Sequence[int] = (),
        gamma: float = 0.1,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.alpha = float(alpha)
        self.upper = float(upper)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.milestones = tuple(int(value) for value in milestones)
        self.gamma = float(gamma)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None
        self._effective_config: Dict[str, Any] = {}

    def fit(self, defense_input: DefenseInput) -> "RelaxLossDefense":
        if defense_input.model_factory is None:
            raise ValueError("RelaxLossDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("RelaxLossDefense requires train_data and train_labels.")

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

        self.training_history = []
        start = time.time()
        for epoch in range(config["epochs"]):
            model.train()
            objective_sum = 0.0
            ce_sum = 0.0
            correct = 0
            count = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = extract_logits(model(batch_x))
                objective = relax_loss_objective(
                    logits,
                    batch_y,
                    epoch=epoch,
                    alpha=config["alpha"],
                    upper=config["upper"],
                )
                mean_ce = F.cross_entropy(logits, batch_y)

                optimizer.zero_grad()
                objective.backward()
                optimizer.step()

                size = len(batch_y)
                objective_sum += float(objective.detach().item()) * size
                ce_sum += float(mean_ce.detach().item()) * size
                correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
                count += size

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "objective": objective_sum / max(count, 1),
                    "cross_entropy": ce_sum / max(count, 1),
                    "train_accuracy": correct / max(count, 1),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
            )
            scheduler.step()

        self.last_runtime_seconds = time.time() - start
        self.defended_model = model.eval()
        self._effective_config = config
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("RelaxLossDefense must be fitted before infer().")

        protected_outputs = None
        if defense_input.samples is not None:
            protected_outputs = predict_labels(
                self.defended_model,
                defense_input.samples,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.defended_model,
            protected_outputs=protected_outputs,
            artifacts={
                "training_history": list(self.training_history),
                "relax_loss_config": dict(self._effective_config),
            },
            intermediate_outputs={"training_history": list(self.training_history)},
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
        model = defense_output.defended_model
        if model is None:
            raise ValueError("No defended model available for evaluation.")
        batch_size = self._effective_config["batch_size"]
        utility: Dict[str, float] = {}
        privacy: Dict[str, float] = {}

        if defense_input.train_data is not None and defense_input.train_labels is not None:
            metrics = classifier_metrics(
                model,
                defense_input.train_data,
                defense_input.train_labels,
                device=self.device,
                batch_size=batch_size,
            )
            utility.update({f"train_{key}": value for key, value in metrics.items()})
        if defense_input.test_data is not None and defense_input.test_labels is not None:
            metrics = classifier_metrics(
                model,
                defense_input.test_data,
                defense_input.test_labels,
                device=self.device,
                batch_size=batch_size,
            )
            utility.update({f"test_{key}": value for key, value in metrics.items()})
            if defense_input.train_data is not None and defense_input.train_labels is not None:
                privacy.update(
                    loss_mia_metrics(
                        model,
                        defense_input.train_data,
                        defense_input.train_labels,
                        defense_input.test_data,
                        defense_input.test_labels,
                        device=self.device,
                        batch_size=batch_size,
                    )
                )

        efficiency = (
            {"train_time": float(self.last_runtime_seconds)}
            if self.last_runtime_seconds is not None
            else None
        )
        return DefenseEvaluationResult(
            utility_metrics=utility or None,
            privacy_metrics=privacy or None,
            efficiency_metrics=efficiency,
            extra_metrics={"training_history": list(self.training_history)},
        )

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "alpha": self.alpha,
            "upper": self.upper,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "milestones": self.milestones,
            "gamma": self.gamma,
            "optimizer": "sgd",
        }
        config.update(dict(overrides))
        config["batch_size"] = int(config["batch_size"])
        config["epochs"] = int(config["epochs"])
        config["learning_rate"] = float(config["learning_rate"])
        config["alpha"] = float(config["alpha"])
        config["upper"] = float(config["upper"])
        config["momentum"] = float(config["momentum"])
        config["weight_decay"] = float(config["weight_decay"])
        config["milestones"] = tuple(int(value) for value in config["milestones"])
        config["gamma"] = float(config["gamma"])
        config["optimizer"] = str(config["optimizer"]).lower()
        if config["batch_size"] <= 0 or config["epochs"] <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if config["alpha"] < 0.0:
            raise ValueError("alpha must be non-negative.")
        if not 0.0 < config["upper"] <= 1.0:
            raise ValueError("upper must be in the interval (0, 1].")
        return config

    def _make_optimizer(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        if config["optimizer"] == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config["weight_decay"],
            )
        if config["optimizer"] != "sgd":
            raise ValueError("optimizer must be either 'sgd' or 'adam'.")
        return torch.optim.SGD(
            model.parameters(),
            lr=config["learning_rate"],
            momentum=config["momentum"],
            weight_decay=config["weight_decay"],
        )


__all__ = ["RelaxLossDefense", "relax_loss_objective"]
