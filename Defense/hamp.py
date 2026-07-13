"""HAMP defense for PyTorch classifiers.

Reference: Chen and Pattabiraman, "Overconfidence is a Dangerous Thing:
Mitigating Membership Inference Attacks by Enforcing Less Confident
Prediction", NDSS 2024.

HAMP has two parts.  During training it uses high-entropy soft targets plus an
entropy reward.  At inference it replaces each logit's magnitude with the
corresponding order statistic from a random non-member input, preserving the
complete class ranking while hiding confidence-based membership signals.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

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
    predict_logits,
    probability_entropy,
    resolve_device,
    to_feature_tensor,
)
from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


def high_entropy_target(num_classes: int, entropy_percentile: float) -> Tuple[float, float]:
    """Return true-class and other-class probabilities at the HAMP entropy target."""
    if num_classes < 2:
        raise ValueError("HAMP requires at least two classes.")
    if not 0.0 <= entropy_percentile <= 1.0:
        raise ValueError("entropy_percentile must be in the interval [0, 1].")

    target_entropy = entropy_percentile * math.log(num_classes)
    if entropy_percentile >= 1.0:
        return 1.0 / num_classes, 1.0 / num_classes
    if entropy_percentile <= 0.0:
        return 1.0, 0.0

    def entropy(top_probability: float) -> float:
        other = (1.0 - top_probability) / (num_classes - 1)
        result = -top_probability * math.log(max(top_probability, 1e-15))
        if other > 0.0:
            result -= (num_classes - 1) * other * math.log(other)
        return result

    low = 1.0 / num_classes
    high = 1.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if entropy(middle) > target_entropy:
            low = middle
        else:
            high = middle
    top_probability = 0.5 * (low + high)
    return top_probability, (1.0 - top_probability) / (num_classes - 1)


def rank_preserving_replacement(
    original_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """Replace values using reference order statistics while preserving ranks."""
    if original_logits.ndim != 2 or reference_logits.ndim != 2:
        raise ValueError("Both logits tensors must have shape (batch, classes).")
    if original_logits.shape[1] != reference_logits.shape[1]:
        raise ValueError("Original and reference logits must have the same class count.")
    if len(reference_logits) == 0:
        raise ValueError("reference_logits must not be empty.")

    if len(reference_logits) != len(original_logits):
        indices = torch.arange(len(original_logits), device=reference_logits.device)
        reference_logits = reference_logits[indices % len(reference_logits)]
    reference_logits = reference_logits.to(
        device=original_logits.device, dtype=original_logits.dtype
    )
    original_order = torch.argsort(original_logits, dim=1, stable=True)
    reference_values = torch.sort(reference_logits, dim=1, stable=True).values
    protected = torch.empty_like(original_logits)
    protected.scatter_(1, original_order, reference_values)
    return protected


class _HAMPProtectedPredictor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        reference_provider: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.model = model
        self.reference_provider = reference_provider

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        raw_logits = extract_logits(self.model(samples))
        reference_logits = self.reference_provider(samples)
        return rank_preserving_replacement(raw_logits, reference_logits)


class HAMPDefense(BaseDefense):
    """Hybrid HAMP training and rank-preserving output defense.

    Training requires ``model_factory``, ``train_data``, and ``train_labels``.
    A pre-trained ``target_model`` may instead be supplied to use only HAMP's
    output modification.  A custom random non-member generator can be passed as
    ``auxiliary_data['nonmember_generator']``; it receives and returns a feature
    tensor with the same shape.
    """

    name = "hamp"
    defense_family = "confidence_regularization"
    defense_mode = "hybrid"
    supported_model_types = ["classifier"]
    required_input_keys = ["model_factory + train data, target_model, or signals['logits']"]
    optional_input_keys = [
        "val_data",
        "val_labels",
        "test_data",
        "test_labels",
        "samples",
        "signals.reference_logits",
        "auxiliary_data.nonmember_generator",
        "auxiliary_data.reference_data",
    ]

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 200,
        learning_rate: float = 0.5,
        entropy_percentile: float = 0.95,
        entropy_weight: float = 0.01,
        entropy_penalty: bool = True,
        modify_output: bool = True,
        momentum: float = 0.99,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.entropy_percentile = float(entropy_percentile)
        self.entropy_weight = float(entropy_weight)
        self.entropy_penalty = bool(entropy_penalty)
        self.modify_output = bool(modify_output)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.protected_predictor: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None
        self._effective_config: Dict[str, Any] = {}
        self._target_probabilities: Optional[Tuple[float, float]] = None
        self._defense_input: Optional[DefenseInput] = None
        self._random_generator: Optional[torch.Generator] = None

    def fit(self, defense_input: DefenseInput) -> "HAMPDefense":
        config = self._merge_config(defense_input.defense_config)
        self._effective_config = config
        self._defense_input = defense_input
        self.training_history = []
        self._target_probabilities = None

        can_train = (
            defense_input.train_data is not None
            and defense_input.train_labels is not None
            and (defense_input.model_factory is not None or defense_input.target_model is not None)
            and bool(config["train_model"])
        )
        if can_train:
            if defense_input.model_factory is not None:
                model = make_model(defense_input.model_factory, self.device)
            else:
                if not isinstance(defense_input.target_model, nn.Module):
                    raise TypeError("target_model must be a torch.nn.Module.")
                model = defense_input.target_model.to(self.device)
            self._train(model, defense_input, config)
            self.defended_model = model.eval()
        elif defense_input.target_model is not None:
            if not isinstance(defense_input.target_model, nn.Module):
                raise TypeError("target_model must be a torch.nn.Module.")
            self.defended_model = defense_input.target_model.to(self.device).eval()
        elif not (defense_input.signals and defense_input.signals.get("logits") is not None):
            raise ValueError(
                "HAMPDefense requires training resources, a target_model, or signals['logits']."
            )

        if self.defended_model is not None:
            if config["modify_output"]:
                self.protected_predictor = _HAMPProtectedPredictor(
                    self.defended_model, self._reference_logits
                ).to(self.device).eval()
            else:
                self.protected_predictor = self.defended_model
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        raw_logits: Optional[torch.Tensor] = None
        protected_logits: Optional[torch.Tensor] = None

        if defense_input.signals and defense_input.signals.get("logits") is not None:
            raw_logits = torch.as_tensor(defense_input.signals["logits"], dtype=torch.float32)
            if self._effective_config["modify_output"]:
                reference = self._reference_logits_for_signals(defense_input, raw_logits)
                protected_logits = rank_preserving_replacement(raw_logits, reference)
            else:
                protected_logits = raw_logits
        elif defense_input.samples is not None and self.defended_model is not None:
            raw_logits = predict_logits(
                self.defended_model,
                defense_input.samples,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )
            predictor = self.protected_predictor or self.defended_model
            protected_logits = predict_logits(
                predictor,
                defense_input.samples,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )

        artifacts: Dict[str, Any] = {
            "training_history": list(self.training_history),
            "hamp_config": dict(self._effective_config),
        }
        if self._target_probabilities is not None:
            artifacts.update(
                {
                    "true_class_probability": self._target_probabilities[0],
                    "other_class_probability": self._target_probabilities[1],
                }
            )
        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.protected_predictor,
            protected_outputs=(
                protected_logits.numpy() if protected_logits is not None else None
            ),
            artifacts=artifacts,
            intermediate_outputs={
                "raw_logits": raw_logits.numpy() if raw_logits is not None else None,
                "protected_logits": protected_logits.numpy() if protected_logits is not None else None,
            },
            metadata={
                "defense_name": self.name,
                "defense_family": self.defense_family,
                "defense_mode": self.defense_mode,
                "output_modified": bool(self._effective_config["modify_output"]),
                "protected_output_type": "logits",
            },
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        predictor = defense_output.protected_predictor or defense_output.defended_model
        utility: Dict[str, float] = {}
        privacy: Dict[str, float] = {}
        batch_size = self._effective_config["batch_size"]

        if predictor is not None:
            for prefix, data, labels in (
                ("train", defense_input.train_data, defense_input.train_labels),
                ("test", defense_input.test_data, defense_input.test_labels),
            ):
                if data is not None and labels is not None:
                    metrics = classifier_metrics(
                        predictor,
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
                        predictor,
                        defense_input.train_data,
                        defense_input.train_labels,
                        defense_input.test_data,
                        defense_input.test_labels,
                        device=self.device,
                        batch_size=batch_size,
                    )
                )

        raw = (defense_output.intermediate_outputs or {}).get("raw_logits")
        protected = (defense_output.intermediate_outputs or {}).get("protected_logits")
        if raw is not None and protected is not None:
            raw_tensor = torch.as_tensor(raw)
            protected_tensor = torch.as_tensor(protected)
            utility["ranking_preservation"] = float(
                torch.equal(
                    torch.argsort(raw_tensor, dim=1, stable=True),
                    torch.argsort(protected_tensor, dim=1, stable=True),
                )
            )
            privacy["raw_mean_entropy"] = float(probability_entropy(raw_tensor).mean().item())
            privacy["protected_mean_entropy"] = float(
                probability_entropy(protected_tensor).mean().item()
            )
            privacy["raw_mean_confidence"] = float(
                torch.softmax(raw_tensor, dim=1).max(dim=1).values.mean().item()
            )
            privacy["protected_mean_confidence"] = float(
                torch.softmax(protected_tensor, dim=1).max(dim=1).values.mean().item()
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

    def _train(
        self,
        model: nn.Module,
        defense_input: DefenseInput,
        config: Dict[str, Any],
    ) -> None:
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
        best_val_accuracy = float("-inf")
        start = time.time()

        for epoch in range(config["epochs"]):
            model.train()
            loss_sum = 0.0
            correct = 0
            count = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                logits = extract_logits(model(batch_x))
                if logits.ndim != 2:
                    raise ValueError("Classifier logits must have shape (batch, classes).")
                num_classes = logits.shape[1]
                if self._target_probabilities is None:
                    self._target_probabilities = high_entropy_target(
                        num_classes, config["entropy_percentile"]
                    )
                true_probability, other_probability = self._target_probabilities
                soft_targets = torch.full_like(logits, other_probability)
                soft_targets.scatter_(1, batch_y[:, None], true_probability)
                # The artifact uses PyTorch's legacy default KL reduction,
                # which averages across both the batch and class dimensions.
                loss = F.kl_div(
                    F.log_softmax(logits, dim=1), soft_targets, reduction="none"
                ).mean()
                if config["entropy_penalty"]:
                    loss = loss - config["entropy_weight"] * probability_entropy(logits).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                size = len(batch_y)
                loss_sum += float(loss.detach().item()) * size
                correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
                count += size

            record: Dict[str, float] = {
                "epoch": float(epoch + 1),
                "train_loss": loss_sum / max(count, 1),
                "train_accuracy": correct / max(count, 1),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if defense_input.val_data is not None and defense_input.val_labels is not None:
                metrics = classifier_metrics(
                    model,
                    defense_input.val_data,
                    defense_input.val_labels,
                    device=self.device,
                    batch_size=config["batch_size"],
                )
                record["val_loss"] = metrics["loss"]
                record["val_accuracy"] = metrics["accuracy"]
                if metrics["accuracy"] > best_val_accuracy + config["validation_min_delta"]:
                    best_val_accuracy = metrics["accuracy"]
                    best_state = copy.deepcopy(model.state_dict())
            self.training_history.append(record)
            scheduler.step()

        if best_state is not None and config["restore_best"]:
            model.load_state_dict(best_state)
        self.last_runtime_seconds = time.time() - start

    def _reference_logits(self, samples: torch.Tensor) -> torch.Tensor:
        if self.defended_model is None or self._defense_input is None:
            raise RuntimeError("HAMP reference logits require a fitted target model.")
        auxiliary = self._defense_input.auxiliary_data or {}
        if auxiliary.get("reference_logits") is not None:
            return torch.as_tensor(
                auxiliary["reference_logits"], device=samples.device, dtype=torch.float32
            )

        if auxiliary.get("reference_data") is not None:
            reference_data = to_feature_tensor(auxiliary["reference_data"])
            if len(reference_data) == 0:
                raise ValueError("auxiliary_data['reference_data'] must not be empty.")
            indices = torch.arange(len(samples)) % len(reference_data)
            random_samples = reference_data[indices].to(samples.device)
        else:
            generator = auxiliary.get("nonmember_generator")
            if generator is not None:
                random_samples = generator(samples.detach())
                random_samples = torch.as_tensor(
                    random_samples, device=samples.device, dtype=samples.dtype
                )
            else:
                random_samples = self._random_samples(samples)
        with torch.no_grad():
            return extract_logits(self.defended_model(random_samples))

    def _reference_logits_for_signals(
        self,
        defense_input: DefenseInput,
        original_logits: torch.Tensor,
    ) -> torch.Tensor:
        signals = defense_input.signals or {}
        auxiliary = defense_input.auxiliary_data or {}
        reference = signals.get("reference_logits", auxiliary.get("reference_logits"))
        if reference is not None:
            return torch.as_tensor(reference, dtype=original_logits.dtype)
        if defense_input.samples is not None and self.defended_model is not None:
            samples = to_feature_tensor(defense_input.samples).to(self.device)
            return self._reference_logits(samples).detach().cpu()
        raise ValueError(
            "Signal-only HAMP output modification requires reference_logits or "
            "samples plus a target_model."
        )

    def _random_samples(self, samples: torch.Tensor) -> torch.Tensor:
        config = self._effective_config
        if self._random_generator is None:
            self._random_generator = torch.Generator(device=samples.device)
            self._random_generator.manual_seed(config["seed"])
        mode = config["random_input_mode"]
        random_values = torch.rand(
            samples.shape,
            dtype=samples.dtype,
            device=samples.device,
            generator=self._random_generator,
        )
        if mode == "bernoulli":
            return (random_values < config["bernoulli_probability"]).to(samples.dtype)
        if mode == "uniform":
            return config["random_input_low"] + random_values * (
                config["random_input_high"] - config["random_input_low"]
            )
        if mode == "normal":
            normal = torch.randn(
                samples.shape,
                dtype=samples.dtype,
                device=samples.device,
                generator=self._random_generator,
            )
            return config["random_input_mean"] + config["random_input_std"] * normal
        raise ValueError("random_input_mode must be 'uniform', 'bernoulli', or 'normal'.")

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "entropy_percentile": self.entropy_percentile,
            "entropy_weight": self.entropy_weight,
            "entropy_penalty": self.entropy_penalty,
            "modify_output": self.modify_output,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "optimizer": "sgd",
            "milestones": (),
            "gamma": 0.1,
            "train_model": True,
            "restore_best": True,
            "validation_min_delta": 0.0,
            "seed": 1,
            "random_input_mode": "uniform",
            "random_input_low": 0.0,
            "random_input_high": 1.0,
            "bernoulli_probability": 0.5,
            "random_input_mean": 0.0,
            "random_input_std": 1.0,
        }
        config.update(dict(overrides))
        for key in ("batch_size", "epochs", "seed"):
            config[key] = int(config[key])
        for key in (
            "learning_rate",
            "entropy_percentile",
            "entropy_weight",
            "momentum",
            "weight_decay",
            "gamma",
            "validation_min_delta",
            "random_input_low",
            "random_input_high",
            "bernoulli_probability",
            "random_input_mean",
            "random_input_std",
        ):
            config[key] = float(config[key])
        for key in ("entropy_penalty", "modify_output", "train_model", "restore_best"):
            config[key] = bool(config[key])
        config["milestones"] = tuple(int(value) for value in config["milestones"])
        config["optimizer"] = str(config["optimizer"]).lower()
        config["random_input_mode"] = str(config["random_input_mode"]).lower()
        if config["batch_size"] <= 0 or config["epochs"] <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if not 0.0 <= config["entropy_percentile"] <= 1.0:
            raise ValueError("entropy_percentile must be in the interval [0, 1].")
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


__all__ = [
    "HAMPDefense",
    "high_entropy_target",
    "rank_preserving_replacement",
]
