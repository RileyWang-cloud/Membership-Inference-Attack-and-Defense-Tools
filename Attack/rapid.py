"""
RAPID attack wrapped with the project's minimal attack interface.

Implementation flow
-------------------
1. Train shadow model(s) and build labeled attack training data.
2. Train reference models for difficulty calibration.
3. For each shadow sample, compute:
   - original membership score
   - difficulty-calibrated score
4. Train an MLP scoring model on these 2-D features.
5. For target-model samples, compute the same two scores and infer membership.

Unified convention:
    higher membership score -> more likely member
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from Attack.utils_rapid.rapid_reference_utils import (
    RAPIDReferenceManager,
    compute_model_original_scores,
)


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
    """Minimal base class shared by attack implementations."""

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

        pred_threshold = float(attack_output.metadata.get("pred_threshold", 0.5))
        if attack_output.membership_preds is None:
            y_pred = (y_score >= pred_threshold).astype(np.int64)
        else:
            y_pred = _to_numpy_1d(attack_output.membership_preds).astype(np.int64)

        return EvaluationResult(
            accuracy=float(accuracy_score(y_true, y_pred)),
            auroc=_safe_auroc(y_true, y_score),
            tpr_at_fpr={
                "1%": _tpr_at_fpr(y_true, y_score, 0.01),
                "0.1%": _tpr_at_fpr(y_true, y_score, 0.001),
            },
        )

    def run(self, attack_input: AttackInput) -> AttackOutput:
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


class ScoringMLP(nn.Module):
    """Small MLP taking [original_score, calibrated_score] as input."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class RAPIDAttack(BaseAttack):
    """
    RAPID attack for classification models.

    Required shadow_data fields
    ---------------------------
    Either:
    - precomputed attack-train features:
        "original_scores", "calibrated_scores", "membership_labels"
    Or:
    - "train_X", "train_y", "test_X", "test_y", "model_factory"

    Required reference_data fields
    ------------------------------
    Either:
    - "reference_manager": a prebuilt RAPIDReferenceManager
    Or:
    - "train_X", "train_y", "test_X", "test_y", "model_factory"
    """

    def __init__(
        self,
        data_sizes: Optional[List[int]] = None,
        random_seed_num: int = 5,
        reference_model_number: int = 10,
        batch_size: int = 128,
        scoring_hidden_dim: int = 32,
        scoring_lr: float = 1e-3,
        scoring_epochs: int = 60,
        pred_threshold: float = 0.5,
        device: Optional[str] = None,
    ) -> None:
        self.data_sizes = data_sizes or [128]
        self.random_seed_num = random_seed_num
        self.reference_model_number = reference_model_number
        self.batch_size = batch_size
        self.scoring_hidden_dim = scoring_hidden_dim
        self.scoring_lr = scoring_lr
        self.scoring_epochs = scoring_epochs
        self.pred_threshold = pred_threshold
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.reference_manager: Optional[RAPIDReferenceManager] = None
        self.scoring_model: Optional[ScoringMLP] = None

    def fit(self, attack_input: AttackInput) -> "RAPIDAttack":
        self.reference_manager = self._build_reference_manager(attack_input)
        x_train, y_train = self._build_attack_training_features(attack_input)
        self.scoring_model = self._fit_scoring_model(x_train, y_train)
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.reference_manager is None or self.scoring_model is None:
            raise RuntimeError("RAPIDAttack must be fitted before infer().")

        original_scores, calibrated_scores, reference_mean_scores = self._compute_target_features(attack_input)
        features = np.stack([original_scores, calibrated_scores], axis=1).astype(np.float32)
        scores = self._predict_scoring_model(features)
        pred_threshold = float(attack_input.config.get("pred_threshold", self.pred_threshold))
        preds = (scores >= pred_threshold).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "original_scores": original_scores,
                "calibrated_scores": calibrated_scores,
                "reference_mean_scores": reference_mean_scores,
            },
            metadata={
                "attack_name": "rapid",
                "pred_threshold": pred_threshold,
                "data_sizes": self.data_sizes,
                "random_seed_num": self.random_seed_num,
                "reference_model_number": self.reference_model_number,
            },
        )

    def _build_reference_manager(self, attack_input: AttackInput) -> RAPIDReferenceManager:
        reference_data = attack_input.reference_data
        if reference_data is None:
            raise ValueError("reference_data is required for RAPIDAttack.fit().")

        if "reference_manager" in reference_data:
            return reference_data["reference_manager"]

        required_keys = ["train_X", "train_y", "test_X", "test_y", "model_factory"]
        missing_keys = [key for key in required_keys if key not in reference_data]
        if missing_keys:
            raise ValueError(
                "reference_data must provide either 'reference_manager' or all of "
                f"{required_keys}. Missing: {missing_keys}"
            )

        manager = RAPIDReferenceManager(
            train_X=np.asarray(reference_data["train_X"]),
            train_y=np.asarray(reference_data["train_y"]),
            test_X=np.asarray(reference_data["test_X"]),
            test_y=np.asarray(reference_data["test_y"]),
            model_factory=reference_data["model_factory"],
            train_fn=reference_data.get("train_fn"),
            train_config=reference_data.get("train_config", {}),
            batch_size=int(reference_data.get("batch_size", self.batch_size)),
            device=str(self.device),
        )
        manager.train_reference_models(
            data_sizes=reference_data.get("data_sizes", self.data_sizes),
            random_seed_num=int(reference_data.get("random_seed_num", self.random_seed_num)),
            reference_model_number=int(
                reference_data.get("reference_model_number", self.reference_model_number)
            ),
        )
        return manager

    def _build_attack_training_features(self, attack_input: AttackInput) -> Tuple[np.ndarray, np.ndarray]:
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("shadow_data is required for RAPIDAttack.fit().")

        if {"original_scores", "calibrated_scores", "membership_labels"} <= set(shadow_data.keys()):
            original_scores = _to_numpy_1d(shadow_data["original_scores"]).astype(np.float64)
            calibrated_scores = _to_numpy_1d(shadow_data["calibrated_scores"]).astype(np.float64)
            membership_labels = _to_numpy_1d(shadow_data["membership_labels"]).astype(np.float32)
            features = np.stack([original_scores, calibrated_scores], axis=1).astype(np.float32)
            return features, membership_labels

        required_keys = ["train_X", "train_y", "test_X", "test_y", "model_factory"]
        missing_keys = [key for key in required_keys if key not in shadow_data]
        if missing_keys:
            raise ValueError(
                "shadow_data must provide either precomputed scores or all of "
                f"{required_keys}. Missing: {missing_keys}"
            )

        shadow_model = self._train_shadow_model(shadow_data)

        shadow_train_X = shadow_data["train_X"]
        shadow_train_y = shadow_data["train_y"]
        shadow_test_X = shadow_data["test_X"]
        shadow_test_y = shadow_data["test_y"]

        member_original = self.reference_manager.compute_original_scores(shadow_model, shadow_train_X, shadow_train_y)  # type: ignore[arg-type]
        member_calibrated, _ = self.reference_manager.compute_calibrated_scores(  # type: ignore[union-attr]
            member_original,
            shadow_train_X,
            shadow_train_y,
        )

        nonmember_original = self.reference_manager.compute_original_scores(shadow_model, shadow_test_X, shadow_test_y)  # type: ignore[arg-type]
        nonmember_calibrated, _ = self.reference_manager.compute_calibrated_scores(  # type: ignore[union-attr]
            nonmember_original,
            shadow_test_X,
            shadow_test_y,
        )

        member_features = np.stack([member_original, member_calibrated], axis=1)
        nonmember_features = np.stack([nonmember_original, nonmember_calibrated], axis=1)

        x_train = np.concatenate([member_features, nonmember_features], axis=0).astype(np.float32)
        y_train = np.concatenate(
            [
                np.ones(len(member_features), dtype=np.float32),
                np.zeros(len(nonmember_features), dtype=np.float32),
            ],
            axis=0,
        )
        return x_train, y_train

    def _train_shadow_model(self, shadow_data: Dict[str, Any]) -> nn.Module:
        model_factory = shadow_data["model_factory"]
        train_fn = shadow_data.get("train_fn", _default_train_fn)
        train_config = shadow_data.get("train_config", {})

        model = model_factory().to(self.device)
        train_loader = _build_loader(
            shadow_data["train_X"],
            shadow_data["train_y"],
            batch_size=int(train_config.get("batch_size", self.batch_size)),
            shuffle=True,
        )
        train_fn(model, train_loader, self.device, train_config)
        model.eval()
        return model

    def _fit_scoring_model(self, x_train: np.ndarray, y_train: np.ndarray) -> ScoringMLP:
        x_tensor = torch.tensor(x_train, dtype=torch.float32)
        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = ScoringMLP(input_dim=x_train.shape[1], hidden_dim=self.scoring_hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.scoring_lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.scoring_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def _compute_target_features(self, attack_input: AttackInput) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if attack_input.labels is None:
            raise ValueError("labels are required for RAPIDAttack inference.")

        labels = attack_input.labels
        if attack_input.signals is not None and "original_scores" in attack_input.signals:
            original_scores = _to_numpy_1d(attack_input.signals["original_scores"]).astype(np.float64)
        else:
            if attack_input.target_model is None:
                raise ValueError("target_model is required when signals['original_scores'] is not provided.")
            original_scores = compute_model_original_scores(
                model=attack_input.target_model,
                samples=attack_input.samples,
                labels=labels,
                batch_size=int(attack_input.config.get("batch_size", self.batch_size)),
                device=self.device,
            )

        if attack_input.signals is not None and "calibrated_scores" in attack_input.signals:
            calibrated_scores = _to_numpy_1d(attack_input.signals["calibrated_scores"]).astype(np.float64)
            reference_mean_scores = (
                _to_numpy_1d(attack_input.signals["reference_mean_scores"]).astype(np.float64)
                if "reference_mean_scores" in attack_input.signals
                else original_scores - calibrated_scores
            )
        else:
            calibrated_scores, reference_mean_scores = self.reference_manager.compute_calibrated_scores(  # type: ignore[union-attr]
                original_scores,
                attack_input.samples,
                labels,
            )
        return original_scores, calibrated_scores, reference_mean_scores

    def _predict_scoring_model(self, features: np.ndarray) -> np.ndarray:
        if self.scoring_model is None:
            raise RuntimeError("Scoring model is not trained.")
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.scoring_model.eval()
        with torch.no_grad():
            logits = self.scoring_model(feature_tensor)
            scores = torch.sigmoid(logits).detach().cpu().numpy()
        return scores.astype(np.float64)


def _default_train_fn(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    train_config: Dict[str, Any],
) -> None:
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


def _build_loader(samples: Any, labels: Any, batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    y_tensor = _to_tensor_1d(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().to(torch.float32)
    return torch.as_tensor(value, dtype=torch.float32)


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
    "RAPIDAttack",
]
