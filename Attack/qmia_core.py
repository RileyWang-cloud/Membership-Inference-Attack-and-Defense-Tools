"""
Reference: "Scalable Membership Inference Attacks via Quantile Regression"
"""

import numpy as np
import torch
import torch.nn.functional as F


def to_onehot(labels, num_classes):
    return F.one_hot(labels, num_classes=num_classes)


# ============================================================================
# Loss Functions
# ============================================================================

def pinball_loss_fn(score, target, quantile):
    """Pinball loss for quantile regression."""
    target = target.reshape([-1, 1])
    assert score.ndim == 2, f"Expected 2d input, got {score.shape}"
    delta_score = target - score
    loss = F.relu(delta_score) * quantile + F.relu(-delta_score) * (1.0 - quantile)
    return loss


def gaussian_loss_fn(score, target, quantile):
    """Gaussian negative log-likelihood loss."""
    assert score.ndim == 2 and score.shape[-1] == 2, f"Expected Nx2 input, got {score.shape}"
    assert target.ndim == 1, f"Expected 1-d target, got {target.shape}"

    mu = score[:, 0]
    log_std = score[:, 1]
    assert mu.shape == log_std.shape and mu.shape == target.shape

    loss = log_std + 0.5 * torch.exp(-2 * log_std) * (target - mu) ** 2
    assert target.shape == loss.shape
    return loss


# ============================================================================
# Score Functions
# ============================================================================

def label_logit_and_hinge_scoring_fn(samples, label, base_model):
    """
    Compute hinge loss score: z_y(x) - max_{y'!=y} z_{y'}(x)

    Returns: (score, logits) where score shape is (n,), logits shape is (n, num_classes)
    """
    base_model.eval()
    with torch.no_grad():
        logits = base_model(samples)
        oh_label = to_onehot(label, logits.shape[-1]).bool()
        score = logits[oh_label]
        score -= torch.max(logits[~oh_label].view(logits.shape[0], -1), dim=1)[0]
        assert score.ndim == 1
    return score, logits


# ============================================================================
# Quantile Rearrangement
# ============================================================================

def rearrange_quantile_fn(test_preds, all_quantiles, target_quantiles=None):
    """
    Rearrange quantiles to ensure monotonicity.

    Based on: Chernozhukov et al. "Quantile and probability curves without crossing." (2010)
    """
    if not target_quantiles:
        target_quantiles = all_quantiles

    scaling = all_quantiles[-1] - all_quantiles[0]
    rescaled_target_qs = (target_quantiles - all_quantiles[0]) / scaling
    q_fixed = torch.quantile(test_preds, rescaled_target_qs, interpolation="linear", dim=-1).T
    assert q_fixed.shape[0] == test_preds.shape[0] and q_fixed.ndim == test_preds.ndim
    return q_fixed


# ============================================================================
# Evaluation Metrics
# ============================================================================

def get_rates(private_target_scores, public_target_scores,
              private_thresholds, public_thresholds):
    """Calculate TPR, TNR, and precision for all thresholds."""
    assert len(private_target_scores.shape) == 1
    assert len(public_target_scores.shape) == 1
    assert len(private_thresholds.shape) == 2
    assert len(public_thresholds.shape) == 2

    prior = 0.0
    tp = (private_target_scores.reshape([-1, 1]) >= private_thresholds).sum(0) + prior
    fn = (private_target_scores.reshape([-1, 1]) < private_thresholds).sum(0) + prior
    tn = (public_target_scores.reshape([-1, 1]) < public_thresholds).sum(0) + prior
    fp = (public_target_scores.reshape([-1, 1]) >= public_thresholds).sum(0) + prior

    tpr = np.nan_to_num(tp / (tp + fn))
    tnr = np.nan_to_num(tn / (tn + fp))
    precision = np.nan_to_num(tpr / (tpr + 1 - tnr))

    return precision, tpr, tnr


def pinball_loss_np(target, score, quantile):
    """NumPy version of pinball loss."""
    target = target.reshape([-1, 1])
    assert score.ndim == 2
    delta_score = target - score
    loss = np.maximum(delta_score * quantile, -delta_score * (1.0 - quantile)).mean(0)
    return loss


# ============================================================================
# QMIA Model
# ============================================================================

class QMIAModel(torch.nn.Module):
    """Quantile Membership Inference Attack Model."""

    def __init__(self, architecture, base_model, num_base_classes,
                 low_quantile=-4, high_quantile=0, n_quantile=41,
                 use_logscale=True, use_gaussian=False,
                 use_target_dependent_scoring=False, use_target_inputs=False,
                 hidden_dims=[512, 512], device='cuda'):
        super().__init__()

        self.base_model = base_model
        self.num_base_classes = num_base_classes
        self.use_gaussian = use_gaussian
        self.use_target_dependent_scoring = use_target_dependent_scoring
        self.use_target_inputs = use_target_inputs
        self.device = device

        # Freeze base model
        for parameter in self.base_model.parameters():
            parameter.requires_grad = False

        # Setup quantile levels
        if use_logscale:
            self.quantile = torch.sort(
                1 - torch.logspace(low_quantile, high_quantile, n_quantile)
            )[0].reshape([1, -1])
        else:
            self.quantile = torch.sort(
                torch.linspace(low_quantile, high_quantile, n_quantile)
            )[0].reshape([1, -1])

        # Setup loss and scoring functions
        self.loss_fn = gaussian_loss_fn if use_gaussian else pinball_loss_fn
        self.target_scoring_fn = label_logit_and_hinge_scoring_fn

    def compute_scores(self, samples, labels):
        """Compute hinge loss scores from base model."""
        scores, _ = self.target_scoring_fn(samples, labels, self.base_model)
        return scores

    def evaluate_attack(self, private_scores, public_scores,
                       private_predicted_thresholds, public_predicted_thresholds):
        """Evaluate attack performance, return metrics dict."""
        precision, tpr, tnr = get_rates(
            private_scores, public_scores,
            private_predicted_thresholds, public_predicted_thresholds
        )

        idx_1pc = np.argmin(np.abs(tnr - 0.99))
        idx_01pc = np.argmin(np.abs(tnr - 0.999))

        return {
            'precision': precision, 'tpr': tpr, 'tnr': tnr,
            'auc': np.abs(np.trapz(tpr, x=1 - tnr)),
            'precision_at_1pct_fpr': precision[idx_1pc],
            'tpr_at_1pct_fpr': tpr[idx_1pc],
            'precision_at_01pct_fpr': precision[idx_01pc],
            'tpr_at_01pct_fpr': tpr[idx_01pc],
        }


# ============================================================================
# Utilities
# ============================================================================

def create_quantile_levels(low_quantile, high_quantile, n_quantile, use_logscale=True):
    """Create quantile levels for training. Returns shape (1, n_quantile)."""
    if use_logscale:
        quantile = torch.sort(
            1 - torch.logspace(low_quantile, high_quantile, n_quantile)
        )[0].reshape([1, -1])
    else:
        quantile = torch.sort(
            torch.linspace(low_quantile, high_quantile, n_quantile)
        )[0].reshape([1, -1])
    return quantile


def print_attack_results(results):
    """Print attack evaluation results."""
    print(f"AUC: {results['auc']:.4f}")
    print(f"@ 1% FPR: Precision={results['precision_at_1pct_fpr']*100:.2f}%, "
          f"TPR={results['tpr_at_1pct_fpr']*100:.2f}%")
    print(f"@ 0.1% FPR: Precision={results['precision_at_01pct_fpr']*100:.2f}%, "
          f"TPR={results['tpr_at_01pct_fpr']*100:.2f}%")


if __name__ == "__main__":
    print("QMIA Core Methods Module")
    print("=" * 50)
    print("Core methods: pinball_loss_fn, gaussian_loss_fn,")
    print("label_logit_and_hinge_scoring_fn, rearrange_quantile_fn,")
    print("get_rates, QMIAModel")
    print("\nUsage:")
    print("  from qmia_core import QMIAModel")
    print("  model = QMIAModel(...)")
    print("  results = model.evaluate_attack(...)")
