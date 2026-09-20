"""
Classic shadow-based membership inference attack.

This module implements the supervised attack proposed by:
Shokri et al., "Membership Inference Attacks Against Machine Learning Models" (2017).

The implementation follows the project's minimal attack interface:
    AttackInput -> AttackOutput

Supported workflow
------------------
1. Train multiple shadow models on their own member splits.
2. Query each shadow model on:
   - its member samples
   - its non-member samples
3. Use these model outputs to build supervised attack training data.
4. Train one binary attack model per class, or one global binary attack model.
5. Query the target model on the target samples and infer membership.

This implementation is intentionally focused on classification models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class AttackInput:
    target_model: Optional[Any]
    samples: Any
    labels: Optional[Any] = None
    membership_labels: Optional[Any] = None
    signals: Optional[Dict[str, Any]] = None
    reference_data: Optional[Dict[str, Any]] = None
    shadow_data: Optional[Dict[str, Any]] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    tpr_at_fpr: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AttackOutput:
    membership_scores: Any
    membership_preds: Optional[Any] = None
    evaluation: Optional[EvaluationResult] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack:
    """Minimal base class from the project interface design."""

    def fit(self, attack_input: AttackInput) -> "BaseAttack":
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        raise NotImplementedError

    def evaluate(
        self,
        attack_output: AttackOutput,
        attack_input: AttackInput,
    ) -> EvaluationResult:
        y_true = _to_numpy_1d(attack_input.membership_labels)
        y_score = _to_numpy_1d(attack_output.membership_scores)

        if attack_output.membership_preds is None:
            y_pred = (y_score >= 0.5).astype(np.int64)
        else:
            y_pred = _to_numpy_1d(attack_output.membership_preds).astype(np.int64)

        metrics = EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auroc=_safe_auroc(y_true, y_score),
            tpr_at_fpr={
                "1%": _tpr_at_fpr(y_true, y_score, 0.01),
                "0.1%": _tpr_at_fpr(y_true, y_score, 0.001),
            },
        )
        return metrics

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


class AttackMLP(nn.Module):
    """Binary attack classifier used on top of shadow model outputs."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ShadowBasedAttack(BaseAttack):
    """
    Classic shadow-model based MIA for classification models.

    Required fields
    ---------------
    - attack_input.target_model:
        trained target classification model, unless signals["probabilities"] is given
    - attack_input.samples:
        target samples to attack
    - attack_input.labels:
        true class labels for target samples
    - attack_input.shadow_data:
        either precomputed shadow outputs or the information needed to train shadow models

    Supported shadow_data formats
    -----------------------------
    Format A: train shadow models inside this class
    {
        "model_factory": callable that returns a new untrained torch.nn.Module,
        "shadow_splits": [
            {
                "train_samples": Tensor,
                "train_labels": Tensor,
                "test_samples": Tensor,
                "test_labels": Tensor,
            },
            ...
        ],
        "train_fn": optional callable(model, train_loader, device, train_config)
    }

    Format B: use precomputed shadow outputs
    {
        "member_outputs": [ndarray or Tensor, ...] or ndarray,
        "member_labels": [ndarray or Tensor, ...] or ndarray,
        "nonmember_outputs": [ndarray or Tensor, ...] or ndarray,
        "nonmember_labels": [ndarray or Tensor, ...] or ndarray,
    }
    """

    def __init__(
        self,
        batch_size: int = 128,
        attack_hidden_dim: int = 64,
        attack_lr: float = 1e-3,
        attack_epochs: int = 50,
        per_class: bool = True,
        device: Optional[str] = None,
    ) -> None:
        self.batch_size = batch_size
        self.attack_hidden_dim = attack_hidden_dim
        self.attack_lr = attack_lr
        self.attack_epochs = attack_epochs
        self.per_class = per_class
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.attack_models: Dict[Any, AttackMLP] = {}
        self.num_classes: Optional[int] = None
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "ShadowBasedAttack":
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("shadow_data is required for ShadowBasedAttack.fit().")

        member_outputs, member_labels, nonmember_outputs, nonmember_labels = (
            self._collect_shadow_attack_data(shadow_data, attack_input.config)
        )
        self.num_classes = self._infer_num_classes(
            attack_input.config,
            member_outputs,
            nonmember_outputs,
            member_labels,
            nonmember_labels,
        )

        self.attack_models = self._train_attack_models(
            member_outputs=member_outputs,
            member_labels=member_labels,
            nonmember_outputs=nonmember_outputs,
            nonmember_labels=nonmember_labels,
        )
        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted:
            raise RuntimeError("ShadowBasedAttack must be fitted before infer().")

        labels = _to_tensor_1d(attack_input.labels, dtype=torch.long)
        if labels is None:
            raise ValueError("labels are required for ShadowBasedAttack inference.")

        target_outputs = self._get_target_probabilities(attack_input)
        features = self._build_attack_features(target_outputs)

        scores = np.zeros(features.shape[0], dtype=np.float32)
        preds = np.zeros(features.shape[0], dtype=np.int64)

        if self.per_class:
            for class_idx in range(self.num_classes): # type: ignore
                class_mask = (labels.cpu().numpy() == class_idx)
                if not np.any(class_mask):
                    continue
                if class_idx not in self.attack_models:
                    raise RuntimeError(f"No attack model found for class {class_idx}.")

                class_features = torch.tensor(features[class_mask], dtype=torch.float32, device=self.device)
                class_scores = self._predict_attack_scores(self.attack_models[class_idx], class_features)
                scores[class_mask] = class_scores
                preds[class_mask] = (class_scores >= 0.5).astype(np.int64)
        else:
            global_model = self.attack_models["global"]
            feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            scores = self._predict_attack_scores(global_model, feature_tensor)
            preds = (scores >= 0.5).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={"target_probabilities": target_outputs},
            metadata={
                "attack_name": "shadow_based",
                "per_class": self.per_class,
                "num_classes": self.num_classes,
            },
        )

    def _collect_shadow_attack_data(
        self,
        shadow_data: Dict[str, Any],
        config: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if "member_outputs" in shadow_data:
            return (
                _stack_feature_arrays(shadow_data["member_outputs"]),
                _stack_label_arrays(shadow_data["member_labels"]).astype(np.int64),
                _stack_feature_arrays(shadow_data["nonmember_outputs"]),
                _stack_label_arrays(shadow_data["nonmember_labels"]).astype(np.int64),
            )

        if "model_factory" not in shadow_data or "shadow_splits" not in shadow_data:
            raise ValueError(
                "shadow_data must provide either precomputed outputs or "
                "'model_factory' with 'shadow_splits'."
            )

        model_factory = shadow_data["model_factory"]
        train_fn = shadow_data.get("train_fn", _default_train_fn)
        train_config = shadow_data.get("train_config", {})
        shadow_splits = shadow_data["shadow_splits"]

        member_outputs, member_labels = [], []
        nonmember_outputs, nonmember_labels = [], []

        for split_idx, split in enumerate(shadow_splits):
            model = model_factory()
            model.to(self.device)

            train_loader = _build_loader(
                split["train_samples"],
                split["train_labels"],
                batch_size=train_config.get("batch_size", self.batch_size),
                shuffle=True,
            )
            train_fn(model, train_loader, self.device, train_config)

            train_probs = _predict_probabilities(
                model,
                split["train_samples"],
                batch_size=self.batch_size,
                device=self.device,
            )
            test_probs = _predict_probabilities(
                model,
                split["test_samples"],
                batch_size=self.batch_size,
                device=self.device,
            )

            member_outputs.append(train_probs)
            member_labels.append(_to_numpy_1d(split["train_labels"]).astype(np.int64))
            nonmember_outputs.append(test_probs)
            nonmember_labels.append(_to_numpy_1d(split["test_labels"]).astype(np.int64))

            _ = split_idx  # keep explicit loop variable for future logging/debugging

        return (
            np.concatenate(member_outputs, axis=0),
            np.concatenate(member_labels, axis=0),
            np.concatenate(nonmember_outputs, axis=0),
            np.concatenate(nonmember_labels, axis=0),
        )

    def _infer_num_classes(
        self,
        config: Dict[str, Any],
        member_outputs: np.ndarray,
        nonmember_outputs: np.ndarray,
        member_labels: np.ndarray,
        nonmember_labels: np.ndarray,
    ) -> int:
        if "num_classes" in config:
            return int(config["num_classes"])
        if member_outputs.ndim == 2:
            return int(member_outputs.shape[1])
        if nonmember_outputs.ndim == 2:
            return int(nonmember_outputs.shape[1])
        return int(max(member_labels.max(), nonmember_labels.max()) + 1)

    def _train_attack_models(
        self,
        member_outputs: np.ndarray,
        member_labels: np.ndarray,
        nonmember_outputs: np.ndarray,
        nonmember_labels: np.ndarray,
    ) -> Dict[Any, AttackMLP]:
        models: Dict[Any, AttackMLP] = {}
        member_features = self._build_attack_features(member_outputs)
        nonmember_features = self._build_attack_features(nonmember_outputs)

        if self.per_class:
            for class_idx in range(self.num_classes): # type: ignore
                member_mask = member_labels == class_idx
                nonmember_mask = nonmember_labels == class_idx
                if not np.any(member_mask) or not np.any(nonmember_mask):
                    continue

                x = np.concatenate(
                    [member_features[member_mask], nonmember_features[nonmember_mask]],
                    axis=0,
                )
                y = np.concatenate(
                    [
                        np.ones(member_mask.sum(), dtype=np.float32),
                        np.zeros(nonmember_mask.sum(), dtype=np.float32),
                    ],
                    axis=0,
                )
                models[class_idx] = self._fit_binary_attack_model(x, y)
        else:
            x = np.concatenate([member_features, nonmember_features], axis=0)
            y = np.concatenate(
                [
                    np.ones(member_features.shape[0], dtype=np.float32),
                    np.zeros(nonmember_features.shape[0], dtype=np.float32),
                ],
                axis=0,
            )
            models["global"] = self._fit_binary_attack_model(x, y)

        if not models:
            raise RuntimeError("No valid attack model could be trained from the given shadow data.")
        return models

    def _fit_binary_attack_model(self, x: np.ndarray, y: np.ndarray) -> AttackMLP:
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = AttackMLP(input_dim=x.shape[1], hidden_dim=self.attack_hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.attack_lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.attack_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def _get_target_probabilities(self, attack_input: AttackInput) -> np.ndarray:
        if attack_input.signals is not None and "probabilities" in attack_input.signals:
            return _to_numpy_2d(attack_input.signals["probabilities"])

        if attack_input.target_model is None:
            raise ValueError(
                "target_model is required when signals['probabilities'] is not provided."
            )

        return _predict_probabilities(
            attack_input.target_model,
            attack_input.samples,
            batch_size=attack_input.config.get("batch_size", self.batch_size),
            device=self.device,
        )

    def _build_attack_features(self, probabilities: np.ndarray) -> np.ndarray:
        if probabilities.ndim == 1:
            probabilities = probabilities[:, None]
        return probabilities.astype(np.float32)

    def _predict_attack_scores(self, model: AttackMLP, features: torch.Tensor) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            logits = model(features)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        return scores.astype(np.float32)


def _default_train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    train_config: Dict[str, Any],
) -> None:
    """Default classifier training loop for shadow models."""
    epochs = int(train_config.get("epochs", 20))
    lr = float(train_config.get("lr", 1e-3))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()


def _predict_probabilities(
    model: nn.Module,
    samples: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _build_loader(samples, labels=None, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch_x = batch[0]
            else:
                batch_x = batch
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu())

    return torch.cat(all_probs, dim=0).numpy().astype(np.float32)


def _build_loader(
    samples: Any,
    labels: Optional[Any],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    sample_tensor = _to_tensor_any(samples)
    if labels is None:
        dataset = TensorDataset(sample_tensor)
    else:
        label_tensor = _to_tensor_1d(labels, dtype=torch.long)
        dataset = TensorDataset(sample_tensor, label_tensor) # type: ignore
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _to_numpy_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _stack_feature_arrays(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        arrays = [_to_numpy_2d(v) for v in value]
        return np.concatenate(arrays, axis=0)
    return _to_numpy_2d(value)


def _stack_label_arrays(value: Any) -> np.ndarray:
    if isinstance(value, (list, tuple)):
        arrays = [_to_numpy_1d(v) for v in value]
        return np.concatenate(arrays, axis=0)
    return _to_numpy_1d(value)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    fpr, tpr, _ = roc_curve(y_true, y_score)
    within_budget = fpr <= fpr_threshold
    if not within_budget.any():
        return 0.0
    return float(tpr[within_budget].max())


__all__ = [
    "AttackInput",
    "AttackOutput",
    "EvaluationResult",
    "BaseAttack",
    "ShadowBasedAttack",
]
