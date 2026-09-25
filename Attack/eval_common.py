"""Unified attack evaluation (benchmark plan v2 §14.1, work item W5).

All benchmark runners compute attack metrics *only* through this module;
per-attack ``evaluate`` copies are never invoked (W5). Protocol:

- ``membership_scores`` convention: higher -> more likely member.
- TPR@x%FPR := highest TPR achievable while FPR <= x% (``Attack/base.py``
  semantics, not the nearest-ROC-point variant).
- Attack Accuracy uses the shadow-calibrated threshold: scores are
  linearly standardized with the shadow model's member/non-member score
  means,
      s' = (s - mu_nm) / (mu_m - mu_nm)
  so that shadow members map to 1 and shadow non-members to 0, and the
  decision threshold is the fixed 0.5. mu_m / mu_nm come from the shared
  Shadow Bundle (never from the evaluation set).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score

from Attack.base import _tpr_at_fpr

IMAGE_FPR_BUDGETS = {"1%": 0.01, "0.1%": 0.001, "0%": 0.0}


def shadow_standardized_accuracy(
    scores: np.ndarray,
    membership_labels: np.ndarray,
    shadow_member_scores: np.ndarray,
    shadow_nonmember_scores: np.ndarray,
) -> Optional[float]:
    """Attack Accuracy with the §14.1 shadow-calibrated 0.5 threshold."""
    mu_m = float(np.mean(shadow_member_scores))
    mu_nm = float(np.mean(shadow_nonmember_scores))
    if mu_m == mu_nm:  # degenerate calibration; report None rather than a fake number
        return None
    standardized = (np.asarray(scores, dtype=np.float64) - mu_nm) / (mu_m - mu_nm)
    preds = (standardized >= 0.5).astype(np.int64)
    return float(np.mean(preds == np.asarray(membership_labels).astype(np.int64)))


def evaluate_scores(
    scores: np.ndarray,
    membership_labels: np.ndarray,
    shadow_member_scores: Optional[np.ndarray] = None,
    shadow_nonmember_scores: Optional[np.ndarray] = None,
    *,
    image_dataset: bool = True,
) -> Dict[str, Optional[float]]:
    """Compute the unified metric block for one attack run (§14.1)."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    membership_labels = np.asarray(membership_labels).astype(np.int64).reshape(-1)
    result: Dict[str, Optional[float]] = {
        "member_count": int((membership_labels == 1).sum()),
        "nonmember_count": int((membership_labels == 0).sum()),
    }
    if len(np.unique(membership_labels)) < 2:
        result["auroc"] = None
    else:
        result["auroc"] = float(roc_auc_score(membership_labels, scores))
    budgets = IMAGE_FPR_BUDGETS if image_dataset else {"1%": 0.01}
    for name, budget in budgets.items():
        key = {"1%": "tpr_at_1pct_fpr", "0.1%": "tpr_at_0_1pct_fpr", "0%": "tpr_at_0pct_fpr"}[name]
        if len(np.unique(membership_labels)) < 2:
            result[key] = None
        else:
            result[key] = float(_tpr_at_fpr(membership_labels, scores, budget))
    if shadow_member_scores is None or shadow_nonmember_scores is None:
        result["attack_accuracy"] = None
    else:
        result["attack_accuracy"] = shadow_standardized_accuracy(
            scores, membership_labels, shadow_member_scores, shadow_nonmember_scores
        )
    return result


__all__ = [
    "evaluate_scores",
    "shadow_standardized_accuracy",
    "IMAGE_FPR_BUDGETS",
]
