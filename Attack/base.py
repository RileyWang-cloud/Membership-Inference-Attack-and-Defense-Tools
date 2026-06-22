"""
Base attack abstractions for MIA attack implementations.

This module turns the design in `ATTACK_INTERFACE_DESIGN_ZH.md` into concrete
Python dataclasses and an abstract base class that other attacks can inherit.

The shared contract (design doc §3, §9):
    attack = SomeAttack(...)
    output = attack.run(attack_input)
        # run() = fit() -> infer() -> evaluate() when labels are available

Mirrors the structure of `Defense/base.py` so the Attack and Defense sides of
the toolkit stay symmetric.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve


@dataclass
class AttackInput:
    """Standardized input for all attack methods (design doc §4, §5).

    `target_model` and `samples` mirror the design doc's positional fields;
    every other field is optional. Each attack documents which fields it
    requires and validates them inside `fit` / `infer`.
    """

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
    """Unified evaluation structure for attack outputs (design doc §8)."""

    accuracy: Optional[float] = None
    auroc: Optional[float] = None
    tpr_at_fpr: Optional[Dict[str, float]] = None
    extra_metrics: Optional[Dict[str, Any]] = None


@dataclass
class AttackOutput:
    """Standardized output for all attack methods (design doc §6, §7).

    Convention (design doc §7.1, §10.3): `membership_scores` must satisfy
    "higher score -> more likely member"; flip the sign inside the attack if
    the raw signal points the other way.
    """

    membership_scores: Any
    membership_preds: Optional[Any] = None
    evaluation: Optional[EvaluationResult] = None
    intermediate_outputs: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAttack(ABC):
    """
    Abstract base class for attack implementations (design doc §3).

    Subclasses must implement `infer`. Training-based attacks usually override
    `fit`; signal-only attacks can keep the default no-op. `evaluate` and `run`
    are shared so that every attack exposes the same call surface.
    """

    def fit(self, attack_input: AttackInput) -> "BaseAttack":
        return self

    @abstractmethod
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        raise NotImplementedError

    def evaluate(
        self,
        attack_output: AttackOutput,
        attack_input: AttackInput,
    ) -> EvaluationResult:
        """Compute accuracy / AUROC / TPR@low-FPR from scores (design doc §8).

        Uses `membership_preds` when the attack provides them, otherwise derives
        hard labels from `membership_scores >= 0` (matches the "higher is
        member" convention with a zero threshold).
        """
        y_true = _to_numpy_1d(attack_input.membership_labels)
        y_score = _to_numpy_1d(attack_output.membership_scores)

        if attack_output.membership_preds is None:
            y_pred = (y_score >= 0.0).astype(np.int64)
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
        """Unified entry point (design doc §9): fit -> infer -> maybe evaluate."""
        self.fit(attack_input)
        output = self.infer(attack_input)
        if attack_input.membership_labels is not None:
            output.evaluation = self.evaluate(output, attack_input)
        return output


def _to_numpy_1d(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("Expected non-empty value for conversion to numpy.")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    """AUROC is undefined when only one class is present (design doc §8.2)."""
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    """Highest TPR achievable while keeping FPR at or below ``fpr_threshold``.

    This is the standard "TPR @ low FPR" privacy metric used in the MIA
    literature (rather than the TPR at the single ROC point whose FPR is
    nearest the threshold, which is undefined on perfectly-separable curves
    where multiple points share FPR = 0).
    """
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
]
