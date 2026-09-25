"""W5：统一攻击评估口径（方案 §14.1）。

所有攻击输出 membership_score（越大越可能 member），本模块是唯一的指标实现：
- AUROC：sklearn
- TPR@x%FPR：FPR ≤ 预算约束下的最大 TPR（Attack/base.py 语义，非"最近 ROC 点"）
- Attack Accuracy：shadow 校准阈值——
    s' = (s − μ_nm) / (μ_m − μ_nm)，μ_m/μ_nm 为 shadow model 在
    shadow_train / shadow_test 上的同型攻击分数均值，s' ≥ 0.5 判 member
- TPR@0.1%FPR 与 TPR@0%FPR 仅图像数据集报告（表格记 N/A）
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_budget: float) -> float:
    """FPR ≤ fpr_budget 约束下的最大 TPR。"""
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, scores)
    mask = fpr <= fpr_budget + 1e-12
    if not mask.any():
        return 0.0
    return float(tpr[mask].max())


def shadow_calibrated_accuracy(
    scores: np.ndarray,
    labels: np.ndarray,
    shadow_member_scores: np.ndarray,
    shadow_nonmember_scores: np.ndarray,
) -> Dict[str, Any]:
    """§14.1 规则 2：线性标准化到 shadow 分数空间后取固定阈值 0.5。

    NaN 鲁棒：LiRA offline 的 shadow 侧可能出现无 OUT 观测的样本（4 ref 全 IN），
    均值估计与判定均跳过 NaN。
    """
    scores = np.asarray(scores, dtype=np.float64)
    sm = np.asarray(shadow_member_scores, dtype=np.float64)
    sn = np.asarray(shadow_nonmember_scores, dtype=np.float64)
    mu_m = float(np.nanmean(sm)) if np.any(~np.isnan(sm)) else 0.0
    mu_nm = float(np.nanmean(sn)) if np.any(~np.isnan(sn)) else 0.0
    denom = mu_m - mu_nm
    valid = ~np.isnan(scores)
    if abs(denom) < 1e-12:
        thr = float(np.nanmedian(np.concatenate([sm, sn])))
        preds = scores >= thr
        return {
            "accuracy": float((preds[valid] == np.asarray(labels)[valid]).mean()) if valid.any() else float("nan"),
            "mu_m": mu_m, "mu_nm": mu_nm, "degenerate": True,
        }
    standardized = (scores - mu_nm) / denom
    preds = standardized >= 0.5
    return {
        "accuracy": float((preds[valid] == np.asarray(labels)[valid]).mean()) if valid.any() else float("nan"),
        "mu_m": mu_m, "mu_nm": mu_nm, "degenerate": False,
    }


def evaluate_attack(
    scores: np.ndarray,
    labels: np.ndarray,
    shadow_member_scores: np.ndarray,
    shadow_nonmember_scores: np.ndarray,
    is_image: bool,
) -> Dict[str, Any]:
    """统一指标出口。labels: 1=member, 0=non-member。"""
    out: Dict[str, Any] = {
        "auroc": auroc(scores, labels),
        **{k: v for k, v in shadow_calibrated_accuracy(scores, labels, shadow_member_scores, shadow_nonmember_scores).items()},
        "tpr_at_1pct_fpr": tpr_at_fpr(scores, labels, 0.01),
    }
    if is_image:
        out["tpr_at_0_1pct_fpr"] = tpr_at_fpr(scores, labels, 0.001)
        out["tpr_at_0pct_fpr"] = tpr_at_fpr(scores, labels, 0.0)
    else:
        out["tpr_at_0_1pct_fpr"] = None  # 表格数据集 N/A（§14.1 规则 3）
        out["tpr_at_0pct_fpr"] = None
    out["member_count"] = int(np.sum(np.asarray(labels) == 1))
    out["nonmember_count"] = int(np.sum(np.asarray(labels) == 0))
    return out
