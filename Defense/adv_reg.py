"""Adversarial-regularization defense for PyTorch classifiers.

Reference: Nasr, Shokri, and Houmansadr, "Machine Learning with Membership
Privacy using Adversarial Regularization", ACM CCS 2018.

The target classifier and a membership discriminator are optimized
alternately.  The discriminator learns from target-training members and an
auxiliary non-member set; the classifier minimizes task loss while suppressing
the discriminator's member confidence.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


class MembershipAdversary(nn.Module):
    """Default AdvReg discriminator over classifier logits and one-hot labels."""

    def __init__(self, num_classes: int, hidden_size: int = 128) -> None:
        super().__init__()
        branch_size = max(32, hidden_size // 2)
        self.logit_branch = nn.Sequential(
            nn.Linear(num_classes, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, branch_size),
            nn.ReLU(),
        )
        self.label_branch = nn.Sequential(
            nn.Linear(num_classes, branch_size),
            nn.ReLU(),
        )
        self.combine = nn.Sequential(
            nn.Linear(2 * branch_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, logits: torch.Tensor, one_hot_labels: torch.Tensor) -> torch.Tensor:
        logit_features = self.logit_branch(logits)
        label_features = self.label_branch(one_hot_labels)
        return self.combine(torch.cat([logit_features, label_features], dim=1)).reshape(-1)


class AdvRegDefense(BaseDefense):
    """Training-time membership privacy through adversarial regularization.

    In addition to the usual model factory and target training set, this method
    requires a disjoint pseudo-non-member set in ``auxiliary_data`` under
    ``nonmember_data`` and ``nonmember_labels`` (the aliases ``attack_data`` and
    ``attack_labels`` are also accepted).
    """

    name = "adv_reg"
    defense_family = "adversarial_regularization"
    defense_mode = "training_time"
    supported_model_types = ["classifier"]
    required_input_keys = [
        "model_factory",
        "train_data",
        "train_labels",
        "auxiliary_data.nonmember_data",
        "auxiliary_data.nonmember_labels",
    ]
    optional_input_keys = [
        "auxiliary_data.adversary_factory",
        "test_data",
        "test_labels",
        "samples",
        "labels",
    ]

    def __init__(
        self,
        batch_size: int = 128,
        epochs: int = 50,
        learning_rate: float = 1e-3,
        adversary_learning_rate: float = 1e-3,
        alpha: float = 1.0,
        adversary_steps: int = 5,
        warmup_epochs: int = 1,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.adversary_learning_rate = float(adversary_learning_rate)
        self.alpha = float(alpha)
        self.adversary_steps = int(adversary_steps)
        self.warmup_epochs = int(warmup_epochs)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.adversary: Optional[nn.Module] = None
        self.training_history: List[Dict[str, float]] = []
        self.last_runtime_seconds: Optional[float] = None
        self._effective_config: Dict[str, Any] = {}
        self._nonmember_data: Any = None
        self._nonmember_labels: Any = None

    def fit(self, defense_input: DefenseInput) -> "AdvRegDefense":
        if defense_input.model_factory is None:
            raise ValueError("AdvRegDefense requires defense_input.model_factory.")
        if defense_input.train_data is None or defense_input.train_labels is None:
            raise ValueError("AdvRegDefense requires train_data and train_labels.")
        auxiliary = defense_input.auxiliary_data or {}
        self._nonmember_data = auxiliary.get("nonmember_data", auxiliary.get("attack_data"))
        self._nonmember_labels = auxiliary.get(
            "nonmember_labels", auxiliary.get("attack_labels")
        )
        if self._nonmember_data is None or self._nonmember_labels is None:
            raise ValueError(
                "AdvRegDefense requires auxiliary_data['nonmember_data'] and "
                "auxiliary_data['nonmember_labels']."
            )

        config = self._merge_config(defense_input.defense_config)
        model = make_model(defense_input.model_factory, self.device)
        member_loader = build_loader(
            defense_input.train_data,
            defense_input.train_labels,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        nonmember_loader = build_loader(
            self._nonmember_data,
            self._nonmember_labels,
            batch_size=config["batch_size"],
            shuffle=True,
        )
        num_classes = self._infer_num_classes(model, member_loader)
        adversary = self._make_adversary(auxiliary.get("adversary_factory"), num_classes, config)
        target_optimizer = self._make_target_optimizer(model, config)
        adversary_optimizer = torch.optim.Adam(
            adversary.parameters(),
            lr=config["adversary_learning_rate"],
            weight_decay=config["adversary_weight_decay"],
        )

        self.training_history = []
        start = time.time()
        for epoch in range(config["epochs"]):
            if epoch < config["warmup_epochs"]:
                classifier_loss, train_accuracy = self._train_classifier_epoch(
                    model, member_loader, target_optimizer
                )
                adversary_loss = 0.0
                adversary_accuracy = 0.0
                for _ in range(config["adversary_steps"]):
                    adversary_loss, adversary_accuracy = self._train_adversary_epoch(
                        model,
                        adversary,
                        member_loader,
                        nonmember_loader,
                        adversary_optimizer,
                        config,
                    )
            else:
                (
                    classifier_loss,
                    train_accuracy,
                    adversary_loss,
                    adversary_accuracy,
                ) = self._train_private_epoch(
                    model,
                    adversary,
                    member_loader,
                    nonmember_loader,
                    target_optimizer,
                    adversary_optimizer,
                    config,
                )

            self.training_history.append(
                {
                    "epoch": float(epoch + 1),
                    "classifier_loss": classifier_loss,
                    "train_accuracy": train_accuracy,
                    "adversary_loss": adversary_loss,
                    "adversary_accuracy": adversary_accuracy,
                }
            )

        self.last_runtime_seconds = time.time() - start
        self.defended_model = model.eval()
        self.adversary = adversary.eval()
        self._effective_config = config
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self.defended_model is None or self.adversary is None:
            raise RuntimeError("AdvRegDefense must be fitted before infer().")
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
                "membership_adversary": self.adversary,
                "training_history": list(self.training_history),
                "adv_reg_config": dict(self._effective_config),
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

        for prefix, data, labels in (
            ("train", defense_input.train_data, defense_input.train_labels),
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
        if self.adversary is not None:
            privacy["internal_adversary_accuracy"] = self._adversary_accuracy(
                model,
                self.adversary,
                defense_input.train_data,
                defense_input.train_labels,
                self._nonmember_data,
                self._nonmember_labels,
                batch_size,
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

    def _train_classifier_epoch(
        self,
        model: nn.Module,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
    ) -> Tuple[float, float]:
        model.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            logits = extract_logits(model(batch_x))
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            size = len(batch_y)
            loss_sum += float(loss.detach().item()) * size
            correct += int((logits.detach().argmax(dim=1) == batch_y).sum().item())
            count += size
        return loss_sum / max(count, 1), correct / max(count, 1)

    def _train_adversary_epoch(
        self,
        model: nn.Module,
        adversary: nn.Module,
        member_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        nonmember_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> Tuple[float, float]:
        model.eval()
        adversary.train()
        loss_sum = 0.0
        correct = 0
        count = 0
        for member_batch, nonmember_batch in zip(member_loader, nonmember_loader):
            loss, batch_correct, size = self._adversary_step(
                model, adversary, member_batch, nonmember_batch, optimizer, config
            )
            loss_sum += loss * size
            correct += batch_correct
            count += size
        return loss_sum / max(count, 1), correct / max(count, 1)

    def _train_private_epoch(
        self,
        model: nn.Module,
        adversary: nn.Module,
        member_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        nonmember_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
        target_optimizer: torch.optim.Optimizer,
        adversary_optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> Tuple[float, float, float, float]:
        nonmember_iterator = iter(nonmember_loader)
        classifier_loss_sum = 0.0
        train_correct = 0
        train_count = 0
        adversary_loss_sum = 0.0
        adversary_correct = 0
        adversary_count = 0

        for member_batch in member_loader:
            try:
                nonmember_batch = next(nonmember_iterator)
            except StopIteration:
                nonmember_iterator = iter(nonmember_loader)
                nonmember_batch = next(nonmember_iterator)

            for _ in range(config["adversary_steps"]):
                loss, batch_correct, size = self._adversary_step(
                    model,
                    adversary,
                    member_batch,
                    nonmember_batch,
                    adversary_optimizer,
                    config,
                )
                adversary_loss_sum += loss * size
                adversary_correct += batch_correct
                adversary_count += size

            model.train()
            adversary.eval()
            for parameter in adversary.parameters():
                parameter.requires_grad_(False)
            member_x, member_y = member_batch
            member_x = member_x.to(self.device)
            member_y = member_y.to(self.device)
            member_logits = extract_logits(model(member_x))
            one_hot = F.one_hot(member_y, num_classes=member_logits.shape[1]).to(
                member_logits.dtype
            )
            membership_confidence = adversary(member_logits, one_hot).reshape(-1)
            task_loss = F.cross_entropy(member_logits, member_y)
            privacy_regularizer = membership_confidence.mean() - 0.5
            classifier_loss = task_loss + config["alpha"] * privacy_regularizer
            target_optimizer.zero_grad()
            classifier_loss.backward()
            target_optimizer.step()
            for parameter in adversary.parameters():
                parameter.requires_grad_(True)

            size = len(member_y)
            classifier_loss_sum += float(classifier_loss.detach().item()) * size
            train_correct += int(
                (member_logits.detach().argmax(dim=1) == member_y).sum().item()
            )
            train_count += size

        return (
            classifier_loss_sum / max(train_count, 1),
            train_correct / max(train_count, 1),
            adversary_loss_sum / max(adversary_count, 1),
            adversary_correct / max(adversary_count, 1),
        )

    def _adversary_step(
        self,
        model: nn.Module,
        adversary: nn.Module,
        member_batch: Tuple[torch.Tensor, torch.Tensor],
        nonmember_batch: Tuple[torch.Tensor, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
    ) -> Tuple[float, int, int]:
        model.eval()
        adversary.train()
        member_x, member_y = (value.to(self.device) for value in member_batch)
        nonmember_x, nonmember_y = (value.to(self.device) for value in nonmember_batch)
        with torch.no_grad():
            member_logits = extract_logits(model(member_x))
            nonmember_logits = extract_logits(model(nonmember_x))
        logits = torch.cat([member_logits, nonmember_logits], dim=0)
        labels = torch.cat([member_y, nonmember_y], dim=0)
        one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype)
        membership = torch.cat(
            [
                torch.ones(len(member_y), device=self.device),
                torch.zeros(len(nonmember_y), device=self.device),
            ]
        )
        probabilities = adversary(logits.detach(), one_hot).reshape(-1)
        loss = self._membership_loss(probabilities, membership, config)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        correct = int(((probabilities.detach() >= 0.5) == membership.bool()).sum().item())
        return float(loss.detach().item()), correct, len(membership)

    def _adversary_accuracy(
        self,
        model: nn.Module,
        adversary: nn.Module,
        member_data: Any,
        member_labels: Any,
        nonmember_data: Any,
        nonmember_labels: Any,
        batch_size: int,
    ) -> float:
        if any(
            value is None
            for value in (member_data, member_labels, nonmember_data, nonmember_labels)
        ):
            return 0.0
        member_loader = build_loader(
            member_data, member_labels, batch_size=batch_size, shuffle=False
        )
        nonmember_loader = build_loader(
            nonmember_data, nonmember_labels, batch_size=batch_size, shuffle=False
        )
        model.eval()
        adversary.eval()
        correct = 0
        count = 0
        with torch.no_grad():
            for member_batch, nonmember_batch in zip(member_loader, nonmember_loader):
                member_x, member_y = (value.to(self.device) for value in member_batch)
                nonmember_x, nonmember_y = (value.to(self.device) for value in nonmember_batch)
                member_logits = extract_logits(model(member_x))
                nonmember_logits = extract_logits(model(nonmember_x))
                logits = torch.cat([member_logits, nonmember_logits])
                labels = torch.cat([member_y, nonmember_y])
                one_hot = F.one_hot(labels, num_classes=logits.shape[1]).to(logits.dtype)
                membership = torch.cat(
                    [
                        torch.ones(len(member_y), device=self.device),
                        torch.zeros(len(nonmember_y), device=self.device),
                    ]
                )
                probabilities = adversary(logits, one_hot).reshape(-1)
                correct += int(((probabilities >= 0.5) == membership.bool()).sum().item())
                count += len(membership)
        return correct / max(count, 1)

    def _infer_num_classes(
        self,
        model: nn.Module,
        loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    ) -> int:
        batch_x, _ = next(iter(loader))
        model.eval()
        with torch.no_grad():
            logits = extract_logits(model(batch_x.to(self.device)))
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("AdvReg requires multi-class logits with shape (batch, classes).")
        return int(logits.shape[1])

    def _make_adversary(
        self,
        factory: Any,
        num_classes: int,
        config: Dict[str, Any],
    ) -> nn.Module:
        if factory is None:
            adversary = MembershipAdversary(num_classes, config["adversary_hidden_size"])
        else:
            try:
                adversary = factory(num_classes)
            except TypeError:
                adversary = factory()
        if not isinstance(adversary, nn.Module):
            raise TypeError("adversary_factory must return a torch.nn.Module.")
        return adversary.to(self.device)

    def _membership_loss(
        self,
        probabilities: torch.Tensor,
        membership: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        if config["adversary_loss"] == "bce":
            return F.binary_cross_entropy(probabilities, membership)
        return F.mse_loss(probabilities, membership)

    def _make_target_optimizer(
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

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "adversary_learning_rate": self.adversary_learning_rate,
            "alpha": self.alpha,
            "adversary_steps": self.adversary_steps,
            "warmup_epochs": self.warmup_epochs,
            "adversary_hidden_size": 128,
            "adversary_loss": "mse",
            "optimizer": "adam",
            "momentum": 0.9,
            "weight_decay": 0.0,
            "adversary_weight_decay": 0.0,
        }
        config.update(dict(overrides))
        for key in (
            "batch_size",
            "epochs",
            "adversary_steps",
            "warmup_epochs",
            "adversary_hidden_size",
        ):
            config[key] = int(config[key])
        for key in (
            "learning_rate",
            "adversary_learning_rate",
            "alpha",
            "momentum",
            "weight_decay",
            "adversary_weight_decay",
        ):
            config[key] = float(config[key])
        config["adversary_loss"] = str(config["adversary_loss"]).lower()
        config["optimizer"] = str(config["optimizer"]).lower()
        if config["batch_size"] <= 0 or config["epochs"] <= 0:
            raise ValueError("batch_size and epochs must be positive.")
        if config["adversary_steps"] <= 0:
            raise ValueError("adversary_steps must be positive.")
        if config["alpha"] < 0.0:
            raise ValueError("alpha must be non-negative.")
        if config["adversary_loss"] not in {"mse", "bce"}:
            raise ValueError("adversary_loss must be either 'mse' or 'bce'.")
        return config


__all__ = ["AdvRegDefense", "MembershipAdversary"]
