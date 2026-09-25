"""
Metric-based membership inference attacks.

Wraps the inspire-group/membership-inference-evaluation library (Song et al.,
USENIX Security 2021) behind the project's unified AttackInput/AttackOutput interface.

Implements five lightweight attack methods that use model-output signals
directly as membership scores without training an attack model:
    - loss
    - correctness
    - confidence
    - entropy
    - modified_entropy

Unified convention:
    higher membership score -> more likely member
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, TensorDataset

# Import the original library
_LIB_DIR = Path(__file__).resolve().parent / "metric_based_lib"
if str(_LIB_DIR) not in sys.path:
    sys.path.insert(0, str(_LIB_DIR))
from membership_inference_attacks import black_box_benchmarks


# ============================================================================
# Interface types
# ============================================================================

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
    """Minimal base class."""

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


# ============================================================================
# Loss Attack
# ============================================================================

class LossAttack(BaseAttack):
    """loss-based MIA. Score = -per_sample_loss. Uses shadow data for calibration."""

    def __init__(self, batch_size: int = 128, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        logits = _get_logits(attack_input, self.device, self.batch_size)
        labels = _to_tensor_1d(attack_input.labels, dtype=torch.long)
        losses = F.cross_entropy(logits, labels.to(self.device), reduction="none")
        scores = (-losses).detach().cpu().numpy().astype(np.float64)
        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores, membership_preds=preds,
            intermediate_outputs={"losses": losses.detach().cpu().numpy()},
            metadata={"attack_name": "loss"},
        )


# ============================================================================
# Correctness Attack
# ============================================================================

class CorrectnessAttack(BaseAttack):
    """correctness-based MIA. Score = 1 if correctly predicted."""

    def __init__(self, batch_size: int = 128, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        logits = _get_logits(attack_input, self.device, self.batch_size)
        labels = _to_tensor_1d(attack_input.labels, dtype=torch.long)
        predictions = logits.argmax(dim=1)
        correct = (predictions == labels.to(self.device)).float()
        scores = correct.detach().cpu().numpy().astype(np.float64)
        return AttackOutput(
            membership_scores=scores, membership_preds=scores.astype(np.int64),
            intermediate_outputs={"predictions": predictions.detach().cpu().numpy()},
            metadata={"attack_name": "correctness"},
        )


# ============================================================================
# Confidence Attack
# ============================================================================

class ConfidenceAttack(BaseAttack):
    """confidence-based MIA using the library's per-class threshold approach."""

    def __init__(self, batch_size: int = 128, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._thresholds: Dict[int, float] = {}
        self._num_classes: Optional[int] = None

    def fit(self, attack_input: AttackInput) -> "ConfidenceAttack":
        shadow = attack_input.shadow_data
        if shadow is None:
            return self
        bench = _build_benchmarks_from_shadow(shadow)
        self._num_classes = bench.num_classes
        for c in range(self._num_classes):
            s_tr = bench.s_tr_conf[bench.s_tr_labels == c]
            s_te = bench.s_te_conf[bench.s_te_labels == c]
            if len(s_tr) > 0 and len(s_te) > 0:
                self._thresholds[c] = bench._thre_setting(s_tr, s_te)
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        probs = _get_probabilities(attack_input, self.device, self.batch_size)
        labs = _to_numpy_1d(attack_input.labels).astype(np.int64)
        n = probs.shape[0]
        confidences = probs[torch.arange(n), torch.as_tensor(labs).to(self.device)].detach().cpu().numpy()

        if self._thresholds:
            scores = np.zeros(n, dtype=np.float64)
            for c, thre in self._thresholds.items():
                mask = labs == c
                scores[mask] = (confidences[mask] >= thre).astype(np.float64)
        else:
            scores = confidences.astype(np.float64)

        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores, membership_preds=preds,
            intermediate_outputs={"confidences": confidences},
            metadata={"attack_name": "confidence"},
        )


# ============================================================================
# Entropy Attack
# ============================================================================

class EntropyAttack(BaseAttack):
    """entropy-based MIA using the library's _entr_comp function."""

    def __init__(self, batch_size: int = 128, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._thresholds: Dict[int, float] = {}
        self._num_classes: Optional[int] = None

    def fit(self, attack_input: AttackInput) -> "EntropyAttack":
        shadow = attack_input.shadow_data
        if shadow is None:
            return self
        bench = _build_benchmarks_from_shadow(shadow)
        self._num_classes = bench.num_classes
        for c in range(self._num_classes):
            s_tr = -bench.s_tr_entr[bench.s_tr_labels == c]
            s_te = -bench.s_te_entr[bench.s_te_labels == c]
            if len(s_tr) > 0 and len(s_te) > 0:
                self._thresholds[c] = bench._thre_setting(s_tr, s_te)
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        probs = _get_probabilities(attack_input, self.device, self.batch_size)
        labs = _to_numpy_1d(attack_input.labels).astype(np.int64)
        # Use the library's entropy function via a throwaway instance
        _dummy = _make_dummy_bench()
        entr_values = _dummy._entr_comp(probs.detach().cpu().numpy())
        scores = -entr_values  # negative entropy -> higher = more likely member

        if self._thresholds:
            final = np.zeros(len(scores), dtype=np.float64)
            for c, thre in self._thresholds.items():
                mask = labs == c
                final[mask] = (scores[mask] >= thre).astype(np.float64)
            scores = final

        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores.astype(np.float64), membership_preds=preds,
            intermediate_outputs={"entropies": entr_values},
            metadata={"attack_name": "entropy"},
        )


# ============================================================================
# Modified Entropy Attack
# ============================================================================

class ModifiedEntropyAttack(BaseAttack):
    """modified-entropy MIA using the library's _m_entr_comp function."""

    def __init__(self, batch_size: int = 128, device: Optional[str] = None) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._thresholds: Dict[int, float] = {}
        self._num_classes: Optional[int] = None

    def fit(self, attack_input: AttackInput) -> "ModifiedEntropyAttack":
        shadow = attack_input.shadow_data
        if shadow is None:
            return self
        bench = _build_benchmarks_from_shadow(shadow)
        self._num_classes = bench.num_classes
        for c in range(self._num_classes):
            s_tr = -bench.s_tr_m_entr[bench.s_tr_labels == c]
            s_te = -bench.s_te_m_entr[bench.s_te_labels == c]
            if len(s_tr) > 0 and len(s_te) > 0:
                self._thresholds[c] = bench._thre_setting(s_tr, s_te)
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        probs = _get_probabilities(attack_input, self.device, self.batch_size)
        labs = _to_numpy_1d(attack_input.labels).astype(np.int64)
        # Use the library's modified entropy function via a throwaway instance
        _dummy = _make_dummy_bench()
        m_entr = _dummy._m_entr_comp(probs.detach().cpu().numpy(), labs)
        scores = -m_entr  # negative modified entropy -> higher = more likely member

        if self._thresholds:
            final = np.zeros(len(scores), dtype=np.float64)
            for c, thre in self._thresholds.items():
                mask = labs == c
                final[mask] = (scores[mask] >= thre).astype(np.float64)
            scores = final

        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores.astype(np.float64), membership_preds=preds,
            intermediate_outputs={"modified_entropies": m_entr},
            metadata={"attack_name": "modified_entropy"},
        )


# ============================================================================
# Helpers using the original library
# ============================================================================

def _build_benchmarks_from_shadow(shadow: Dict[str, Any]) -> black_box_benchmarks:
    """Build a black_box_benchmarks instance from shadow_data dict."""
    required = ["s_tr_outputs", "s_tr_labels", "s_te_outputs", "s_te_labels",
                  "t_tr_outputs", "t_tr_labels", "t_te_outputs", "t_te_labels"]
    missing = [k for k in required if k not in shadow]
    if missing:
        raise ValueError(f"shadow_data missing required keys: {missing}")
    return black_box_benchmarks(
        shadow_train_performance=(_to_np(shadow["s_tr_outputs"]), _to_np(shadow["s_tr_labels"])),
        shadow_test_performance=(_to_np(shadow["s_te_outputs"]), _to_np(shadow["s_te_labels"])),
        target_train_performance=(_to_np(shadow["t_tr_outputs"]), _to_np(shadow["t_tr_labels"])),
        target_test_performance=(_to_np(shadow["t_te_outputs"]), _to_np(shadow["t_te_labels"])),
        num_classes=int(shadow.get("num_classes", 2)),
    )


def _make_dummy_bench() -> black_box_benchmarks:
    """Create a throwaway instance to access _entr_comp / _m_entr_comp methods."""
    import gc
    # Use __new__ to skip __init__ since we only need the utility methods
    obj = black_box_benchmarks.__new__(black_box_benchmarks)
    obj.num_classes = 2
    return obj


def _get_logits(input: AttackInput, device: torch.device, batch_size: int) -> torch.Tensor:
    if input.signals and "logits" in input.signals:
        return _to_tensor_2d(input.signals["logits"]).to(device)
    if input.target_model is None:
        raise ValueError("target_model required when signals['logits'] not provided")
    return _predict_logits(input.target_model, input.samples, batch_size, device)


def _get_probabilities(input: AttackInput, device: torch.device, batch_size: int) -> torch.Tensor:
    if input.signals:
        if "probabilities" in input.signals:
            return _to_tensor_2d(input.signals["probabilities"]).to(device)
        if "logits" in input.signals:
            return F.softmax(_to_tensor_2d(input.signals["logits"]).to(device), dim=1)
    if input.target_model is None:
        raise ValueError("target_model required when signals not provided")
    return F.softmax(_predict_logits(input.target_model, input.samples, batch_size, device), dim=1)


def _predict_logits(model: nn.Module, samples: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    loader = _build_loader(samples, None, batch_size, shuffle=False)
    model.eval()
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            all_logits.append(model(batch[0].to(device)).detach().cpu())
    return torch.cat(all_logits, dim=0)


def _build_loader(samples: Any, labels: Optional[Any], batch_size: int, shuffle: bool) -> DataLoader:
    x = _to_tensor_2d(samples)
    ds = TensorDataset(x) if labels is None else TensorDataset(x, _to_tensor_1d(labels, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_2d(value: Any) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.unsqueeze(1).to(torch.float32) if t.dim() == 1 else t.to(torch.float32)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.reshape(-1)
    return t.to(dtype) if dtype is not None else t


def _to_numpy_1d(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy().reshape(-1) if isinstance(value, torch.Tensor) else np.asarray(value).reshape(-1)


def _to_np(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> Optional[float]:
    return float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) >= 2 else None


def _tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    """W5 (benchmark plan §14.1): unified to ``Attack/base.py`` semantics —
    highest TPR achievable while FPR <= budget (not nearest-ROC-point)."""
    from Attack.base import _tpr_at_fpr as _base_tpr_at_fpr

    return _base_tpr_at_fpr(y_true, y_score, fpr_threshold)


__all__ = [
    "AttackInput", "AttackOutput", "EvaluationResult", "BaseAttack",
    "LossAttack", "CorrectnessAttack", "ConfidenceAttack",
    "EntropyAttack", "ModifiedEntropyAttack",
]
