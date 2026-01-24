"""
Core Algorithms for Privacy Auditing and Membership Inference Attacks

Algorithms for:
- Signal computation (softmax, Taylor, soft-margin, log-logit scaling)
- Privacy risk assessment (LiRA, Relative, Reference, Population attacks)

Based on:
- Carlini et al., "Membership Inference Attacks From First Principles"
- Ye et al., "Augmented membership inference attacks"
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import norm, trim_mean
from sklearn.metrics import auc, roc_curve
from typing import Dict, List, Tuple, Optional, Union


class PrivacyAuditor:
    """Privacy auditor using membership inference attacks."""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def convert_signals(self, logits: torch.Tensor, labels: torch.Tensor,
                      signal_type: str, temperature: float = 1.0,
                      extra_params: Optional[Dict] = None) -> torch.Tensor:
        """Convert model logits to attack signals."""
        if signal_type == 'softmax':
            return self._softmax_signal(logits, labels, temperature)
        elif signal_type == 'taylor':
            n = extra_params.get("taylor_n", 3) if extra_params else 3
            return self._taylor_signal(logits, labels, temperature, n)
        elif signal_type == 'soft-margin':
            m = extra_params.get("taylor_m", 1.0) if extra_params else 1.0
            return self._soft_margin_signal(logits, labels, temperature, m)
        elif signal_type == 'taylor-soft-margin':
            m = extra_params.get("taylor_m", 1.0) if extra_params else 1.0
            n = extra_params.get("taylor_n", 3) if extra_params else 3
            return self._taylor_soft_margin_signal(logits, labels, temperature, m, n)
        elif signal_type == 'logits':
            return logits
        elif signal_type == 'log-logit-scaling':
            return self._log_logit_scaling_signal(logits, labels)
        else:
            raise ValueError(f"Unsupported signal type: {signal_type}")

    def _softmax_signal(self, logits: torch.Tensor, labels: torch.Tensor,
                       temperature: float) -> torch.Tensor:
        """Compute softmax probability of correct class."""
        scaled_logits = logits / temperature
        max_logits, _ = torch.max(scaled_logits, dim=1, keepdim=True)
        scaled_logits = scaled_logits - max_logits
        exp_logits = torch.exp(torch.clamp(scaled_logits, max=50, min=-50))
        exp_sum = exp_logits.sum(dim=1, keepdim=True)
        true_exp_logits = exp_logits.gather(1, labels.view(-1, 1))
        return torch.clamp((true_exp_logits / exp_sum).squeeze(), min=0, max=1)

    def _taylor_signal(self, logits: torch.Tensor, labels: torch.Tensor,
                      temperature: float, n: int) -> torch.Tensor:
        """Compute Taylor series approximation of softmax."""
        scaled_logits = logits / temperature
        taylor_exp = self._taylor_expansion(scaled_logits, n)
        taylor_sum = taylor_exp.sum(dim=1, keepdim=True)
        true_taylor = taylor_exp.gather(1, labels.view(-1, 1))
        return (true_taylor / taylor_sum).squeeze()

    def _soft_margin_signal(self, logits: torch.Tensor, labels: torch.Tensor,
                           temperature: float, m: float) -> torch.Tensor:
        """Compute soft-margin probability."""
        scaled_logits = logits / temperature
        exp_logits = torch.exp(scaled_logits)
        exp_sum = exp_logits.sum(dim=1, keepdim=True)
        true_logits = scaled_logits.gather(1, labels.view(-1, 1))
        true_exp_logits = exp_logits.gather(1, labels.view(-1, 1))
        exp_sum = exp_sum - true_exp_logits
        soft_true = torch.exp(true_logits - m)
        exp_sum = exp_sum + soft_true
        return (soft_true / exp_sum).squeeze()

    def _taylor_soft_margin_signal(self, logits: torch.Tensor, labels: torch.Tensor,
                                  temperature: float, m: float, n: int) -> torch.Tensor:
        """Compute Taylor series with soft margin."""
        scaled_logits = logits / temperature
        taylor_logits = self._taylor_expansion(scaled_logits, n)
        taylor_sum = taylor_logits.sum(dim=1, keepdim=True)
        true_logits = scaled_logits.gather(1, labels.view(-1, 1))
        true_taylor = taylor_logits.gather(1, labels.view(-1, 1))
        taylor_sum = taylor_sum - true_taylor
        soft_taylor = self._taylor_expansion(true_logits - m, n)
        taylor_sum = taylor_sum + soft_taylor
        return (soft_taylor / taylor_sum).squeeze()

    def _log_logit_scaling_signal(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """LiRA logit scaling signal."""
        max_logits, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - max_logits
        exp_logits = torch.exp(logits)
        exp_logits = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

        batch_size = logits.shape[0]
        true_probs = exp_logits[torch.arange(batch_size), labels[:batch_size]]
        exp_logits[torch.arange(batch_size), labels[:batch_size]] = 0
        wrong_probs = torch.sum(exp_logits, dim=1)

        return torch.log(true_probs + 1e-45) - torch.log(wrong_probs + 1e-45)

    def _taylor_expansion(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Taylor series expansion of exp(x) up to n terms."""
        result = torch.ones_like(x)
        power = x
        for i in range(1, n):
            result = result + power / self._factorial(i)
            power = power * x
        return result

    def _factorial(self, n: int) -> int:
        fact = 1
        for i in range(2, n + 1):
            fact *= i
        return fact

    def lira_attack(self, target_signals: torch.Tensor,
                   reference_signals: torch.Tensor,
                   membership: torch.Tensor,
                   target_indices: torch.Tensor,
                   fix_variance: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """LiRA (Leave-One-Out Reference Attack)."""
        target_signals = target_signals[target_indices]
        membership = membership[target_indices]

        if len(target_signals.shape) > 1:
            target_signals = target_signals.squeeze()

        num_targets = len(target_indices)
        member_indices = torch.where(membership.bool())[0]
        nonmember_indices = torch.where(~membership.bool())[0]

        # Compute mean_in and mean_out from member/non-member reference signals
        if len(member_indices) > 0:
            mean_in = torch.median(reference_signals[:, member_indices], dim=0).values
            mean_in_full = torch.zeros(reference_signals.shape[1], device=reference_signals.device)
            mean_in_full[member_indices] = mean_in
            mean_in_subset = mean_in_full[target_indices]
        else:
            mean_in_subset = torch.zeros(num_targets, device=reference_signals.device)

        if len(nonmember_indices) > 0:
            mean_out = torch.median(reference_signals[:, nonmember_indices], dim=0).values
            mean_out_full = torch.zeros(reference_signals.shape[1], device=reference_signals.device)
            mean_out_full[nonmember_indices] = mean_out
            mean_out_subset = mean_out_full[target_indices]
        else:
            mean_out_subset = torch.zeros(num_targets, device=reference_signals.device)

        # Compute log-likelihood ratios
        if fix_variance:
            std_in = torch.std(reference_signals) + 1e-30
            std_out = torch.std(reference_signals) + 1e-30
            pr_in = -norm.logpdf(target_signals.cpu().numpy(), mean_in_subset.cpu().numpy(), std_in)
            pr_out = -norm.logpdf(target_signals.cpu().numpy(), mean_out_subset.cpu().numpy(), std_out)
        else:
            if len(member_indices) > 0:
                std_in = torch.std(reference_signals[:, member_indices], dim=0)
                std_in_full = torch.zeros(reference_signals.shape[1], device=reference_signals.device)
                std_in_full[member_indices] = std_in
                std_in_subset = std_in_full[target_indices]
            else:
                std_in_subset = torch.ones(num_targets, device=reference_signals.device)

            if len(nonmember_indices) > 0:
                std_out = torch.std(reference_signals[:, nonmember_indices], dim=0)
                std_out_full = torch.zeros(reference_signals.shape[1], device=reference_signals.device)
                std_out_full[nonmember_indices] = std_out
                std_out_subset = std_out_full[target_indices]
            else:
                std_out_subset = torch.ones(num_targets, device=reference_signals.device)

            pr_in = -norm.logpdf(target_signals.cpu().numpy(), mean_in_subset.cpu().numpy(), std_in_subset.cpu().numpy() + 1e-30)
            pr_out = -norm.logpdf(target_signals.cpu().numpy(), mean_out_subset.cpu().numpy(), std_out_subset.cpu().numpy() + 1e-30)

        scores = pr_in - pr_out
        return scores, membership.cpu().numpy()

    def relative_attack(self, target_signals: torch.Tensor,
                       reference_signals: torch.Tensor,
                       membership: torch.Tensor,
                       target_indices: torch.Tensor,
                       population_indices: torch.Tensor,
                       gamma: float = 1.0,
                       proportiontocut: float = 0.1,
                       temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Relative membership inference attack using population comparison."""
        mean_target = trim_mean(reference_signals[:, target_indices],
                               proportiontocut=proportiontocut, axis=0)
        mean_population = trim_mean(reference_signals[:, population_indices],
                                   proportiontocut=proportiontocut, axis=0)

        if len(target_signals.shape) == 1:
            target_probs = target_signals[target_indices] / (mean_target * temperature + 1e-30)
            pop_probs = target_signals[population_indices] / (mean_population * temperature + 1e-30)
        else:
            target_probs = target_signals[0, target_indices] / (mean_target * temperature + 1e-30)
            pop_probs = target_signals[0, population_indices] / (mean_population * temperature + 1e-30)

        scores = torch.outer(target_probs, 1.0 / pop_probs)
        predictions = -((scores > gamma).float().mean(dim=1)).cpu().numpy()
        answers = membership[target_indices].cpu().numpy()

        return predictions, answers

    def reference_attack(self, target_signals: torch.Tensor,
                        reference_signals: torch.Tensor,
                        membership: torch.Tensor,
                        target_indices: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Reference-based attack using linear interpolation."""
        def signal_to_loss(signal):
            return np.log((1 + np.exp(signal)) / np.exp(signal))

        if len(target_signals.shape) == 1:
            target_losses = signal_to_loss(target_signals[target_indices])
        else:
            target_losses = signal_to_loss(target_signals[0, target_indices])

        reference_losses = signal_to_loss(reference_signals[:, target_indices])

        dummy_min = np.zeros((1, len(target_losses)))
        dummy_max = dummy_min + 1000
        all_losses = np.sort(np.concatenate([reference_losses, dummy_max, dummy_min], axis=0), axis=0)

        discrete_alpha = np.linspace(0, 1, len(all_losses))
        predictions = []

        for i in range(len(target_losses)):
            losses_i = all_losses[:, i]
            prob = np.interp(target_losses[i], losses_i, discrete_alpha)
            predictions.append(prob)

        return np.array(predictions), membership[target_indices].cpu().numpy()

    def population_attack(self, target_signals: torch.Tensor,
                         population_signals: torch.Tensor,
                         membership: torch.Tensor,
                         target_indices: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Population-based attack using empirical CDF."""
        def signal_to_loss(signal):
            return np.log((1 + np.exp(signal)) / np.exp(signal))

        if len(target_signals.shape) == 1:
            target_losses = signal_to_loss(target_signals[target_indices])
        else:
            target_losses = signal_to_loss(target_signals[0, target_indices])

        pop_losses = signal_to_loss(population_signals)

        if len(pop_losses.shape) > 1:
            pop_losses = pop_losses.flatten()

        target_losses_np = np.array(target_losses)
        pop_losses_np = np.array(pop_losses)

        target_losses_reshaped = target_losses_np.reshape(1, -1)
        pop_losses_reshaped = pop_losses_np.reshape(-1, 1)

        cdf = np.mean(pop_losses_reshaped <= target_losses_reshaped, axis=0)

        return cdf, membership[target_indices].cpu().numpy()

    def evaluate_attack(self, predictions: np.ndarray, answers: np.ndarray) -> Dict:
        """Evaluate attack performance using AUC, accuracy, and TPR@FPR."""
        fpr, tpr, thresholds = roc_curve(answers, -predictions)
        roc_auc = auc(fpr, tpr)

        fpr_levels = [0.01, 0.001, 0.0001, 0.00001, 0.0]
        tpr_at_fpr = {}
        threshold_at_fpr = {}

        for fpr_level in fpr_levels:
            idx = np.where(fpr <= fpr_level)[0][-1]
            tpr_at_fpr[fpr_level] = tpr[idx]
            threshold_at_fpr[fpr_level] = thresholds[idx]

        acc = np.max(1 - (fpr + (1 - tpr)) / 2)

        return {
            'auc': roc_auc,
            'accuracy': acc,
            'tpr_at_fpr': tpr_at_fpr,
            'thresholds': threshold_at_fpr,
            'fpr': fpr,
            'tpr': tpr
        }


def load_signals_from_file(file_path: str) -> np.ndarray:
    """Load pre-computed signals from file."""
    try:
        return np.load(file_path)
    except Exception as e:
        raise FileNotFoundError(f"Could not load signals from {file_path}: {e}")


def create_signal_matrix(dataset_size: int, num_augmentations: int = 1,
                        num_queries: int = 1) -> np.ndarray:
    """Create empty signal matrix initialized with NaN."""
    if num_augmentations > 0:
        if num_queries > 1:
            return np.full((dataset_size, num_augmentations, num_queries), np.nan)
        else:
            return np.full((dataset_size, num_augmentations), np.nan)
    else:
        if num_queries > 1:
            return np.full((dataset_size, 1, num_queries), np.nan)
        else:
            return np.full((dataset_size, 1), np.nan)


def get_bins_center(ymedians: List[float]) -> List[float]:
    """Calculate bin centers from medians."""
    length = len(ymedians)
    bins = np.linspace(0, 1, length + 1)
    bin_centers = []

    for i in range(1, length + 1):
        bin_center = (bins[i - 1] + bins[i]) / 2
        bin_centers.append(bin_center)

    return bin_centers


def str2bool(value: Union[str, bool]) -> bool:
    """Convert string to boolean."""
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes", "t")


def example_privacy_audit():
    """Example: privacy audit using relative attack."""
    config = {
        'audit': {
            'signal': 'softmax',
            'temperature': 1.0,
            'gamma': 1.0,
            'proportiontocut': 0.1,
            'fix_variance': False
        }
    }

    auditor = PrivacyAuditor(config)

    batch_size = 100
    num_classes = 10
    num_reference_models = 50

    target_logits = torch.randn(batch_size, num_classes)
    target_labels = torch.randint(0, num_classes, (batch_size,))
    reference_logits = torch.randn(num_reference_models, batch_size, num_classes)
    membership = torch.randint(0, 2, (batch_size,)).bool()

    target_signals = auditor.convert_signals(
        target_logits, target_labels,
        config['audit']['signal'],
        config['audit']['temperature']
    )

    reference_signals = torch.stack([
        auditor.convert_signals(reference_logits[i], target_labels,
                              config['audit']['signal'],
                              config['audit']['temperature'])
        for i in range(num_reference_models)
    ])

    subset_size = min(batch_size // 2, 25)
    target_indices = torch.arange(subset_size)
    population_indices = torch.arange(subset_size, subset_size * 2)

    predictions, answers = auditor.relative_attack(
        target_signals.unsqueeze(0),
        reference_signals,
        membership,
        target_indices,
        population_indices,
        config['audit']['gamma'],
        config['audit']['proportiontocut'],
        config['audit']['temperature']
    )

    results = auditor.evaluate_attack(predictions, answers)

    print(f"Attack AUC: {results['auc']:.4f}")
    print(f"Attack Accuracy: {results['accuracy']:.4f}")
    print(f"TPR@1%FPR: {results['tpr_at_fpr'][0.01]:.4f}")

    return results


if __name__ == "__main__":
    results = example_privacy_audit()
