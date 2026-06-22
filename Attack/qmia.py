"""
Quantile Membership Inference Attack (QMIA), wrapped with the project's minimal
attack interface.

Reimplemented from scratch following ``ATTACK_INTERFACE_DESIGN_ZH.md``; the
earlier ``qmia_core.py`` fragment was removed as unreliable (it defined the
loss / scoring helpers but shipped no quantile-regression head, no training
loop, and no ``forward``).

Reference
---------
Bertran et al., "Scalable Membership Inference Attacks via Quantile Regression"
(2024).

Algorithm
---------
The target classifier is frozen. For any sample ``x`` (with task label ``y``)
we use the confidence-margin score::

    s(x) = z_y(x) - max_{y' != y} z_{y'}(x)

A member is fit tightly by the model, so its clean margin sits in the upper tail
of the margins the model assigns to *augmented* versions of ``x``. We therefore
train a small quantile-regression network ``Q(x)`` that, given the clean ``x``,
predicts several quantiles of the margin distribution over augmentations of
``x``. Training uses the pinball loss on offline (reference) data::

    for each offline x_i, a augmentations x_{i,a}:  target margin s(x_{i,a})
    pinball_k(target, Q_k(x_i), q_k)   summed over quantile levels k

Membership score (design doc §7.1, §10.3: higher -> more likely member)::

    membership_score = s(x) - Q_{alpha*}(x)

where ``alpha*`` is a configurable operating quantile (default 0.9). Predicted
quantiles are passed through Chernozhukov et al. (2010) rearrangement to remove
quantile crossing.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Attack.base import AttackInput, AttackOutput, BaseAttack


class QMIAAttack(BaseAttack):
    """
    Quantile Membership Inference Attack.

    No shadow classifier is trained; the only trained component is the
    quantile-regression head, learned from offline (reference) data.

    Required AttackInput fields
    ---------------------------
    - target_model
        A frozen-capable classifier returning logits of shape ``(N, C)``. It is
        put in eval mode and its gradients are disabled during the attack.
    - samples
        Query samples ``X`` to attack (numpy array or torch tensor).
    - labels
        Task labels ``y`` for the queries (needed for the margin score).
    - shadow_data["fit_X"], shadow_data["fit_y"]
        Offline reference data used to train the quantile-regression head. Must
        be disjoint from the query set in membership terms (e.g. public data).

    Optional
    --------
    - config (overrides constructor defaults at runtime): ``operating_quantile``,
      ``n_augmentations``, ``aug_noise``, ``n_epochs``, ``lr``, ``batch_size``,
      ``calibration_fpr``. Quantile levels and the head architecture are set via
      the constructor only.

    Main output
    -----------
    - ``membership_scores`` (= margin - predicted operating-quantile; higher ->
      more likely member), ``membership_preds`` (query score above the offline
      reference-calibrated threshold; falls back to ``score >= 0`` when
      ``calibration_fpr`` is not in (0, 1)), plus clean margins, predicted
      quantiles and the decision threshold in ``intermediate_outputs``.
    """

    def __init__(
        self,
        quantile_levels: Optional[Sequence[float]] = None,
        hidden_dims: Sequence[int] = (128, 128),
        n_epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        n_augmentations: int = 16,
        aug_noise: float = 0.1,
        operating_quantile: float = 0.9,
        calibration_fpr: float = 0.05,
        device: Optional[str] = None,
    ) -> None:
        self.quantile_levels = (
            torch.as_tensor(np.asarray(quantile_levels, dtype=np.float64))
            if quantile_levels is not None
            else _default_quantile_levels()
        ).float()
        if self.quantile_levels.ndim != 1 or len(self.quantile_levels) < 1:
            raise ValueError("quantile_levels must be a non-empty 1-D sequence in (0, 1).")
        self.quantile_levels, _ = torch.sort(self.quantile_levels)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.n_augmentations = int(n_augmentations)
        self.aug_noise = float(aug_noise)
        self.operating_quantile = float(operating_quantile)
        self.calibration_fpr = float(calibration_fpr)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._quantile_net: Optional[_QuantileNet] = None
        self._offline_threshold: Optional[float] = None

    # ------------------------------------------------------------------- fit
    def fit(self, attack_input: AttackInput) -> "QMIAAttack":
        if attack_input.target_model is None:
            raise ValueError("QMIAAttack.fit requires attack_input.target_model.")
        cfg = attack_input.config
        n_epochs = int(cfg.get("n_epochs", self.n_epochs))
        n_augmentations = int(cfg.get("n_augmentations", self.n_augmentations))
        aug_noise = float(cfg.get("aug_noise", self.aug_noise))
        lr = float(cfg.get("lr", self.lr))
        batch_size = int(cfg.get("batch_size", self.batch_size))
        operating_quantile = float(cfg.get("operating_quantile", self.operating_quantile))
        self.calibration_fpr = float(cfg.get("calibration_fpr", self.calibration_fpr))

        fit_X, fit_y = _require_fit_data(attack_input)
        target_model = _freeze(attack_input.target_model).to(self.device)

        aug_margins = self._augmented_margins(
            target_model, fit_X, fit_y, n_augmentations, aug_noise, batch_size
        ).to(self.device)  # (M, A)
        fit_X_t = _to_tensor_2d(fit_X).to(self.device)
        fit_y_t = _to_tensor_1d(fit_y, torch.long).to(self.device)

        input_dim = int(fit_X_t.shape[1])
        net = _QuantileNet(input_dim, self.hidden_dims, len(self.quantile_levels)).to(self.device)
        optim = torch.optim.Adam(net.parameters(), lr=lr)
        levels = self.quantile_levels.to(self.device)

        net.train()
        n = fit_X_t.shape[0]
        idx = torch.arange(n, device=self.device)
        for _ in range(n_epochs):
            perm = idx[torch.randperm(n, device=self.device)]
            for start in range(0, n, batch_size):
                batch_idx = perm[start : start + batch_size]
                xb = fit_X_t[batch_idx]  # (B, d)  -- clean input to the net
                tb = aug_margins[batch_idx]  # (B, A)
                pred = net(xb)  # (B, K)
                loss = _pinball_loss(pred, tb, levels)
                optim.zero_grad()
                loss.backward()
                optim.step()

        self._quantile_net = net

        # Calibrate the hard-decision threshold on the offline (non-member)
        # data: flag a query as member iff its score exceeds the level that only
        # ``calibration_fpr`` of offline non-members exceed. This is QMIA's
        # intended operating procedure (a reference-calibrated threshold), not a
        # naive ``score >= 0``.
        self._offline_threshold = self._calibrate_threshold(
            net, target_model, fit_X_t, fit_y_t, operating_quantile, batch_size
        )
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self._quantile_net is None:
            raise RuntimeError("QMIAAttack must be fitted before infer().")
        if attack_input.target_model is None:
            raise ValueError("QMIAAttack.infer requires attack_input.target_model.")
        if attack_input.labels is None:
            raise ValueError("QMIAAttack.infer requires attack_input.labels (task labels).")
        if attack_input.samples is None:
            raise ValueError("QMIAAttack.infer requires attack_input.samples.")

        operating_quantile = float(attack_input.config.get("operating_quantile", self.operating_quantile))
        batch_size = int(attack_input.config.get("batch_size", self.batch_size))

        target_model = _freeze(attack_input.target_model).to(self.device)
        X_q = _to_tensor_2d(attack_input.samples).to(self.device)
        y_q = _to_tensor_1d(attack_input.labels, torch.long).to(self.device)

        scores, clean_margin, pred, k_op = self._score_samples(
            self._quantile_net, target_model, X_q, y_q, operating_quantile, batch_size
        )
        threshold = self._offline_threshold
        if threshold is not None:
            preds = (scores > threshold).astype(np.int64)
        else:
            preds = (scores >= 0.0).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "clean_margin": clean_margin,
                "operating_quantile": float(self.quantile_levels[k_op].item()),
                "predicted_quantiles": pred.numpy(),
                "quantile_levels": self.quantile_levels.numpy(),
                "decision_threshold": threshold,
            },
            metadata={
                "attack_name": "qmia",
                "n_augmentations": int(attack_input.config.get("n_augmentations", self.n_augmentations)),
                "aug_noise": float(attack_input.config.get("aug_noise", self.aug_noise)),
                "n_epochs": int(attack_input.config.get("n_epochs", self.n_epochs)),
                "calibration_fpr": self.calibration_fpr,
            },
        )

    # -------------------------------------------------------------- internals
    def _score_samples(
        self,
        net: _QuantileNet,
        target_model: nn.Module,
        X_t: torch.Tensor,
        y_t: torch.Tensor,
        operating_quantile: float,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor, int]:
        """Score samples as ``clean_margin - Q_{operating_quantile}(x)``.

        Returns ``(scores, clean_margin, rearranged_pred, k_op)``.
        """
        clean_margin = _margin_scores(target_model, X_t, y_t, batch_size, self.device)
        net.eval()
        with torch.no_grad():
            pred = _rearrange_quantiles(net(X_t).cpu(), self.quantile_levels)
        k_op = int(torch.argmin(torch.abs(self.quantile_levels - operating_quantile)).item())
        scores = clean_margin - pred[:, k_op].numpy()
        return scores, clean_margin, pred, k_op

    def _calibrate_threshold(
        self,
        net: _QuantileNet,
        target_model: nn.Module,
        fit_X_t: torch.Tensor,
        fit_y_t: torch.Tensor,
        operating_quantile: float,
        batch_size: int,
    ) -> Optional[float]:
        """Offline (non-member) score at the ``(1 - calibration_fpr)`` quantile.

        Returns ``None`` when ``calibration_fpr`` is outside (0, 1), in which
        case ``infer`` falls back to ``score >= 0``.
        """
        fpr = self.calibration_fpr
        if not (0.0 < fpr < 1.0):
            return None
        offline_scores, _, _, _ = self._score_samples(
            net, target_model, fit_X_t, fit_y_t, operating_quantile, batch_size
        )
        return float(np.quantile(offline_scores, 1.0 - fpr))

    @torch.no_grad()
    def _augmented_margins(
        self,
        target_model: nn.Module,
        fit_X: Any,
        fit_y: Any,
        n_augmentations: int,
        aug_noise: float,
        batch_size: int,
    ) -> torch.Tensor:
        """Margins of ``n_augmentations`` noisy copies of each offline sample.

        Returns a CPU tensor of shape ``(M, A)`` where row ``i`` holds the
        margins of the ``A`` augmentations of sample ``i``.
        """
        X = _to_tensor_2d(fit_X).to(self.device)
        y = _to_tensor_1d(fit_y, torch.long).to(self.device)
        per_aug = [
            _margin_scores(
                target_model,
                X + aug_noise * torch.randn_like(X),
                y,
                batch_size,
                self.device,
            )
            for _ in range(n_augmentations)
        ]
        return torch.from_numpy(np.stack(per_aug, axis=1))  # (M, A)


# --------------------------------------------------------------------- helpers
@torch.no_grad()
def _margin_scores(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Confidence margin ``z_y - max_{y' != y} z_y'`` per sample, shape ``(N,)``."""
    model.eval()
    chunks = []
    for start in range(0, X.shape[0], batch_size):
        logits = model(X[start : start + batch_size])  # (B, C)
        num_classes = logits.shape[-1]
        oh = F.one_hot(y[start : start + batch_size], num_classes).bool()
        picked = logits[oh]
        masked = logits.masked_fill(oh, float("-inf"))
        max_other = masked.max(dim=1).values
        chunks.append((picked - max_other).cpu().numpy())
    return np.concatenate(chunks)


def _pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: torch.Tensor) -> torch.Tensor:
    """Sum of per-quantile-level pinball losses.

    ``pred``   : (B, K) predicted quantiles
    ``target`` : (B, A) observed (augmented) margins
    ``quantiles``: (K,) quantile levels in (0, 1)
    """
    delta = target.unsqueeze(1) - pred.unsqueeze(2)  # (B, K, A)
    levels = quantiles.view(1, -1, 1)
    loss = F.relu(delta) * levels + F.relu(-delta) * (1.0 - levels)
    return loss.mean(dim=(0, 2)).sum()


def _rearrange_quantiles(pred: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Chernozhukov et al. (2010) rearrangement to remove quantile crossing.

    ``pred`` : (N, K) predicted quantiles (one row per sample)
    ``levels``: (K,) sorted quantile levels
    Returns (N, K) with each row monotonically non-decreasing.
    """
    span = float(levels[-1] - levels[0])
    if span <= 0:
        return pred
    rescaled = ((levels - levels[0]) / span).clamp(0.0, 1.0)
    fixed = torch.quantile(pred, rescaled, dim=-1)  # (K, N)
    return fixed.T


class _QuantileNet(nn.Module):
    """MLP mapping a sample to ``K`` quantile predictions."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], n_quantiles: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(last, dim))
            layers.append(nn.ReLU())
            last = dim
        layers.append(nn.Linear(last, n_quantiles))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_quantile_levels() -> torch.Tensor:
    """Default QMIA levels: ``1 - logspace(-4, 0, 41)``, sorted ascending."""
    levels = 1.0 - torch.logspace(-4, 0, 41)
    return torch.sort(levels)[0]


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _require_fit_data(attack_input: AttackInput) -> Tuple[Any, Any]:
    shadow = attack_input.shadow_data or {}
    fit_X = shadow.get("fit_X")
    fit_y = shadow.get("fit_y")
    if fit_X is None or fit_y is None:
        raise ValueError(
            "QMIAAttack.fit requires shadow_data['fit_X'] and shadow_data['fit_y'] "
            "(offline reference data to train the quantile-regression head)."
        )
    return fit_X, fit_y


def _to_tensor_2d(value: Any) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.float().reshape(t.shape[0], -1) if t.ndim > 2 else t.float()


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.reshape(-1)
    return t.to(dtype) if dtype is not None else t


__all__ = ["QMIAAttack"]
