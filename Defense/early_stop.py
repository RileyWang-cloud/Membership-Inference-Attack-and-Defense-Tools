"""Early-stopping defense for PyTorch classifiers.

Reference: Rezaei and Liu, "On the Difficulty of Membership Inference Attacks",
CVPR 2021 workshop version / the systematic MIA evaluation released in 2021.

The reference evaluation compares checkpoints at fixed epoch budgets.  This
implementation supports that protocol through ``stop_epoch`` and also supports
the usual validation-monitored early stopping protocol through ``patience``.
"""

from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional, Sequence

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


class EarlyStopDefense(BaseDefense):
    """Stop classifier training before memorization widens the MIA signal.

    Required ``DefenseInput`` fields are ``model_factory``, training data and
    labels, and validation data and labels.  Set ``defense_config['stop_epoch']``
    for a fixed checkpoint protocol; otherwise ``monitor`` and ``patience``
    select and restore the best validation checkpoint.
    """

    name = "early_stop"
    defense_family = "training_strategy"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = [
        "model_factory",
        "train_data",
        "train_labels",
        "val_data",
        "val_labels",
    ]
    optional_input_keys = ["test_data", "test_labels", "samples", "labels"]

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        patience: int = 5,
        min_delta: float = 0.0,
        monitor: str = "val_loss",
        min_epochs: int = 1,
        restore_best: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.min_epochs = int(min_epochs)
        self.restore_best = bool(restore_best)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None
        self.selected_epoch: Optional[int] = None
        self.stopped_epoch: Optional[int] = None
        self.stop_reason: Optional[str] = None
        self._effective_config: Dict[str, Any] = {}

    def fit(self, defense_input: DefenseInput) -> "EarlyStopDefense":
        if defense_input.model_factory is None:
            raise ValueError("EarlyStopDefense requires defense_input.model_factory.")
        required = (
            defense_input.train_data,
            defense_input.train_labels,
            defense_input.val_data,
            defense_input.val_labels,
        )
        if any(value is None for value in required):
            raise ValueError(
                "EarlyStopDefense requires train_data, train_labels, val_data, and val_labels."
            )

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

        best_state: Optional[Dict[str, torch.Tensor]] = None
        best_value = float("inf") if config["monitor"] == "val_loss" else float("-inf")
        best_epoch = 0
        epochs_without_improvement = 0
        max_epochs = config["epochs"]
        if config["stop_epoch"] is not None:
            max_epochs = min(max_epochs, config["stop_epoch"])

        self.training_history = []
        self.stop_reason = "max_epochs"
        start = time.time()
        for epoch in range(max_epochs):
            model.train()
            train_loss_sum = 0.0
            train_correct = 0
            train_count = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = extract_logits(model(batch_x))
                loss = F.cross_entropy(logits, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                size = len(batch_y)
                train_loss_sum += float(loss.detach().item()) * size
                train_correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
                train_count += size

            val_metrics = classifier_metrics(
                model,
                defense_input.val_data,
                defense_input.val_labels,
                device=self.device,
                batch_size=config["batch_size"],
            )
            record = {
                "epoch": float(epoch + 1),
                "train_loss": train_loss_sum / max(train_count, 1),
                "train_accuracy": train_correct / max(train_count, 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            self.training_history.append(record)

            current = record[config["monitor"]]
            improved = self._is_improved(current, best_value, config)
            if improved:
                best_value = current
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            scheduler.step()
            if (
                config["stop_epoch"] is None
                and epoch + 1 >= config["min_epochs"]
                and epochs_without_improvement >= config["patience"]
            ):
                self.stop_reason = "validation_patience"
                break

        self.stopped_epoch = len(self.training_history)
        if config["stop_epoch"] is not None:
            self.stop_reason = "fixed_epoch"
            self.selected_epoch = self.stopped_epoch
        elif config["restore_best"] and best_state is not None:
            model.load_state_dict(best_state)
            self.selected_epoch = best_epoch
        else:
            self.selected_epoch = self.stopped_epoch

        self.last_runtime_seconds = time.time() - start
        self.defended_model = model.eval()
        self._effective_config = config
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None:
            raise RuntimeError("EarlyStopDefense must be fitted before infer().")
        protected_outputs = None
        if defense_input.samples is not None:
            protected_outputs = predict_labels(
                self.defended_model,
                defense_input.samples,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )
        artifacts = {
            "training_history": list(self.training_history),
            "early_stop_config": dict(self._effective_config),
            "selected_epoch": self.selected_epoch,
            "stopped_epoch": self.stopped_epoch,
            "stop_reason": self.stop_reason,
        }
        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.defended_model,
            protected_outputs=protected_outputs,
            artifacts=artifacts,
            intermediate_outputs={"training_history": list(self.training_history)},
            metadata={
                "defense_name": self.name,
                "defense_family": self.defense_family,
                "defense_mode": self.defense_mode,
                "selected_epoch": self.selected_epoch,
                "stopped_epoch": self.stopped_epoch,
                "stop_reason": self.stop_reason,
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
        utility: Dict[str, float] = {
            "selected_epoch": float(self.selected_epoch or 0),
            "stopped_epoch": float(self.stopped_epoch or 0),
        }
        privacy: Dict[str, float] = {}

        for prefix, data, labels in (
            ("train", defense_input.train_data, defense_input.train_labels),
            ("validation", defense_input.val_data, defense_input.val_labels),
            ("test", defense_input.test_data, defense_input.test_labels),
        ):
            if data is not None and labels is not None:
                metrics = classifier_metrics(
                    model,
                    data,
                    labels,
                    device=self.device,
                    batch_size=batch_size,
                )
                utility.update({f"{prefix}_{key}": value for key, value in metrics.items()})
        if (
            defense_input.train_data is not None
            and defense_input.train_labels is not None
            and defense_input.test_data is not None
            and defense_input.test_labels is not None
        ):
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
            utility_metrics=utility,
            privacy_metrics=privacy or None,
            efficiency_metrics=efficiency,
            extra_metrics={
                "stop_reason": self.stop_reason,
                "training_history": list(self.training_history),
            },
        )

    def _is_improved(
        self,
        current: float,
        best: float,
        config: Dict[str, Any],
    ) -> bool:
        if config["monitor"] == "val_loss":
            return current < best - config["min_delta"]
        return current > best + config["min_delta"]

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor": self.monitor,
            "min_epochs": self.min_epochs,
            "restore_best": self.restore_best,
            "stop_epoch": None,
            "optimizer": "adam",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "milestones": (),
            "gamma": 0.1,
        }
        config.update(dict(overrides))
        for key in ("batch_size", "epochs", "patience", "min_epochs"):
            config[key] = int(config[key])
        if config["stop_epoch"] is not None:
            config["stop_epoch"] = int(config["stop_epoch"])
        for key in (
            "learning_rate",
            "min_delta",
            "momentum",
            "weight_decay",
            "gamma",
        ):
            config[key] = float(config[key])
        config["monitor"] = str(config["monitor"]).lower()
        config["optimizer"] = str(config["optimizer"]).lower()
        config["restore_best"] = bool(config["restore_best"])
        config["milestones"] = tuple(int(value) for value in config["milestones"])
        if config["monitor"] not in {"val_loss", "val_accuracy"}:
            raise ValueError("monitor must be 'val_loss' or 'val_accuracy'.")
        if config["batch_size"] <= 0 or config["epochs"] <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if config["patience"] < 1:
            raise ValueError("patience must be at least 1.")
        if config["stop_epoch"] is not None and config["stop_epoch"] <= 0:
            raise ValueError("stop_epoch must be positive when provided.")
        return config

    def _make_optimizer(
        self,
        model: nn.Module,
        config: Dict[str, Any],
    ) -> torch.optim.Optimizer:
        if config["optimizer"] == "sgd":
            return torch.optim.SGD(
                model.parameters(),
                lr=config["learning_rate"],
                momentum=config["momentum"],
                weight_decay=config["weight_decay"],
            )
        if config["optimizer"] != "adam":
            raise ValueError("optimizer must be either 'sgd' or 'adam'.")
        return torch.optim.Adam(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )


__all__ = ["EarlyStopDefense"]
