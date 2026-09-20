"""
RMIA attack wrapped with the project's minimal attack interface.

This implementation follows the reference-model training pattern and the
`compute_rmia_scores` logic from the reference file, but is implemented
independently and packaged behind the unified attack interface.

Unified convention:
    higher membership score -> more likely member
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from Attack.utils_rmia.rmia_reference_utils import RMIAReferenceManager


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


class RMIAAttack(BaseAttack):
    """
    RMIA attack for classification models.

    Required reference_data fields
    ------------------------------
    Either:
    - "reference_manager": a prebuilt RMIAReferenceManager

    Or:
    - "train_X", "train_y", "test_X", "test_y"
    - "model_factory": callable returning a fresh classification model

    Required attack-time metadata
    -----------------------------
    - metadata["sample_indices"]:
        global indices for attacked samples
    - metadata["population_indices"]:
        global indices for population samples

    Required attack-time data for population scoring
    -----------------------------------------------
    One of:
    - reference_data["population_samples"] plus population labels
    - signals["population_probabilities"]
    """

    def __init__(
        self,
        data_sizes: Optional[list[int]] = None,
        random_seed_num: int = 5,
        reference_model_number: int = 10,
        batch_size: int = 128,
        a: float = 0.0,
        gamma: float = 1.0,
        pred_threshold: float = 0.5,
        device: Optional[str] = None,
        comparison_chunk_size: int = 1024,
    ) -> None:
        self.data_sizes = data_sizes or [128]
        self.random_seed_num = random_seed_num
        self.reference_model_number = reference_model_number
        self.batch_size = batch_size
        self.a = a
        self.gamma = gamma
        self.pred_threshold = pred_threshold
        if comparison_chunk_size < 1:
            raise ValueError("comparison_chunk_size must be at least 1.")
        self.comparison_chunk_size = comparison_chunk_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reference_manager: Optional[RMIAReferenceManager] = None

    def fit(self, attack_input: AttackInput) -> "RMIAAttack":
        reference_data = attack_input.reference_data
        if reference_data is None:
            raise ValueError("reference_data is required for RMIAAttack.fit().")

        if "reference_manager" in reference_data:
            self.reference_manager = reference_data["reference_manager"]
            return self

        required_keys = ["train_X", "train_y", "test_X", "test_y", "model_factory"]
        missing_keys = [key for key in required_keys if key not in reference_data]
        if missing_keys:
            raise ValueError(
                "reference_data must provide either 'reference_manager' or all of "
                f"{required_keys}. Missing: {missing_keys}"
            )

        manager = RMIAReferenceManager(
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
        required_parts = []
        for key in ("sample_indices", "population_indices"):
            value = attack_input.metadata.get(key, reference_data.get(key))
            if value is not None:
                required_parts.append(np.asarray(value, dtype=np.int64).reshape(-1))
        required_indices = np.unique(np.concatenate(required_parts)) if required_parts else None
        manager.train_reference_models(
            data_sizes=reference_data.get("data_sizes", self.data_sizes),
            random_seed_num=int(reference_data.get("random_seed_num", self.random_seed_num)),
            reference_model_number=int(
                reference_data.get("reference_model_number", self.reference_model_number)
            ),
            ensure_full_coverage=bool(reference_data.get("ensure_full_coverage", True)),
            min_reference_observations=int(
                reference_data.get("min_reference_observations", 2)
            ),
            required_indices=required_indices,
        )
        self.reference_manager = manager
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.reference_manager is None:
            raise RuntimeError("RMIAAttack must be fitted before infer().")

        sample_indices = self._resolve_indices(attack_input, "sample_indices")
        population_indices = self._resolve_indices(attack_input, "population_indices")
        target_probs = self._get_target_probabilities(attack_input)
        population_probs = self._get_population_probabilities(attack_input)

        true_labels = _to_numpy_1d(attack_input.labels).astype(np.int64) if attack_input.labels is not None else None
        population_labels = self._get_population_labels(attack_input)

        scores, details = self.reference_manager.compute_rmia_scores(
            target_model_preds=target_probs,
            sample_indices=sample_indices,
            population_preds=population_probs,
            population_indices=population_indices,
            true_labels=true_labels,
            population_true_labels=population_labels,
            a=float(attack_input.config.get("a", self.a)),
            gamma=float(attack_input.config.get("gamma", self.gamma)),
            comparison_chunk_size=int(
                attack_input.config.get("comparison_chunk_size", self.comparison_chunk_size)
            ),
            return_details=True,
        )
        pred_threshold = float(attack_input.config.get("pred_threshold", self.pred_threshold))
        preds = (scores >= pred_threshold).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "target_probabilities": target_probs,
                "population_probabilities": population_probs,
                "sample_indices": sample_indices,
                "population_indices": population_indices,
                "valid_reference_mask": details["valid_target_mask"],
                "valid_population_mask": details["valid_population_mask"],
                "reference_coverage": float(np.mean(details["valid_target_mask"])) if len(details["valid_target_mask"]) > 0 else 0.0,
                "population_reference_coverage": float(np.mean(details["valid_population_mask"])) if len(details["valid_population_mask"]) > 0 else 0.0,
                "pr_x": details["pr_x"],
                "pr_z": details["pr_z"],
                "ratio_x": details["ratio_x"],
                "ratio_z": details["ratio_z"],
            },
            metadata={
                "attack_name": "rmia",
                "data_sizes": self.data_sizes,
                "random_seed_num": self.random_seed_num,
                "reference_model_number": self.reference_model_number,
                "a": float(attack_input.config.get("a", self.a)),
                "gamma": float(attack_input.config.get("gamma", self.gamma)),
                "pred_threshold": pred_threshold,
                "comparison_chunk_size": int(
                    attack_input.config.get("comparison_chunk_size", self.comparison_chunk_size)
                ),
            },
        )

    def _resolve_indices(self, attack_input: AttackInput, key: str) -> np.ndarray:
        if key in attack_input.metadata:
            return np.asarray(attack_input.metadata[key], dtype=np.int64)
        if attack_input.reference_data is not None and key in attack_input.reference_data:
            return np.asarray(attack_input.reference_data[key], dtype=np.int64)
        raise ValueError(f"RMIAAttack requires {key} in attack_input.metadata or attack_input.reference_data.")

    def _get_target_probabilities(self, attack_input: AttackInput) -> np.ndarray:
        if attack_input.signals is not None and "probabilities" in attack_input.signals:
            return _to_numpy_2d(attack_input.signals["probabilities"])
        if attack_input.target_model is None:
            raise ValueError("target_model is required when signals['probabilities'] is not provided.")
        return _predict_probabilities(
            model=attack_input.target_model,
            samples=attack_input.samples,
            batch_size=int(attack_input.config.get("batch_size", self.batch_size)),
            device=self.device,
        )

    def _get_population_probabilities(self, attack_input: AttackInput) -> np.ndarray:
        if attack_input.signals is not None and "population_probabilities" in attack_input.signals:
            return _to_numpy_2d(attack_input.signals["population_probabilities"])
        if attack_input.reference_data is None or "population_samples" not in attack_input.reference_data:
            raise ValueError(
                "RMIAAttack requires either signals['population_probabilities'] or "
                "reference_data['population_samples']."
            )
        if attack_input.target_model is None:
            raise ValueError("target_model is required when population probabilities are not precomputed.")
        return _predict_probabilities(
            model=attack_input.target_model,
            samples=attack_input.reference_data["population_samples"],
            batch_size=int(attack_input.config.get("batch_size", self.batch_size)),
            device=self.device,
        )

    def _get_population_labels(self, attack_input: AttackInput) -> Optional[np.ndarray]:
        if attack_input.reference_data is None:
            return None
        if "population_labels" in attack_input.reference_data:
            return _to_numpy_1d(attack_input.reference_data["population_labels"]).astype(np.int64)
        return None


def _predict_probabilities(
    model: nn.Module,
    samples: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _build_loader(samples, labels=None, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu())
    return torch.cat(all_probs, dim=0).numpy().astype(np.float64)


def _build_loader(samples: Any, labels: Optional[Any], batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    if labels is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_tensor = _to_tensor_1d(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
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


def _to_numpy_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


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
    "RMIAAttack",
]
