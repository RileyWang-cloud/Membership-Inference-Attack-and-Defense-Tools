"""
GAN-Leaks membership inference against generative models, wrapped with the
project's minimal attack interface.

Reimplemented from scratch following ``ATTACK_INTERFACE_DESIGN_ZH.md``; the
earlier ``GAN_Leaks_Core.py`` was removed as unreliable (only the FBB k-NN
attack had a body; PBB shipped an ``initialize_z`` stub with no optimisation and
WB was an empty shell).

Reference
---------
Hayes et al., "GAN-Leaks: A Taxonomy of Membership Inference Attacks against
Generative Models" (PoPETs 2023 / AsiaCCS 2020 lineage).

Core idea
---------
A generator reconstructs the samples it was trained on (members) better than
non-members. The membership signal is therefore a *reconstruction error*: a
query with a small reconstruction error is likely a member. To honour the
unified convention (design doc §7.1, §10.3: higher score -> more likely member)
the score is the negated reconstruction error::

    membership_score = -reconstruction_error

Attack modes (by threat-model / access level)
---------------------------------------------
- ``mode="fbb"``  Full Black-Box. Only a batch of generated samples is needed.
  The reconstruction error of a query is its distance to the nearest generated
  samples (k-NN in sample space). No generator gradients required.
- ``mode="pbb"``  Partial Black-Box. The attacker can query the generator
  ``G(z)`` and its gradients. Each query is inverted by optimising a latent
  ``z`` to minimise ``||G(z) - x||^2`` (+ a latent-norm regulariser); the
  residual is the reconstruction error.

Hard decisions (``membership_preds``) use a threshold calibrated on a known
non-member reference set (``reference_data["calibration_X"]``): a query is
flagged as member when its reconstruction error is below the reference
``(1 - calibration_fpr)`` quantile. Without a reference set, preds fall back to
``score >= 0`` and ``membership_scores`` / AUROC remain the primary output.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from Attack.base import AttackInput, AttackOutput, BaseAttack


class GANLeaksAttack(BaseAttack):
    """
    GAN-Leaks reconstruction-based membership inference.

    Required AttackInput fields
    ---------------------------
    - samples
        Query points ``X`` to attack (members + non-members), shape ``(N, D)``
        or ``(N, C, H, W)`` (flattened internally).
    - mode == "fbb": ``signals["generated_samples"]``
        A matrix of pre-generated samples ``(M, D)`` (or images) to build the
        k-NN index over.
    - mode == "pbb": ``target_model``
        A differentiable generator ``G(z)`` returning reconstructions in the
        query space. Its input (latent) dimension must match ``z_dim``.

    Optional
    --------
    - reference_data["calibration_X"]
        Known non-member reference points used to calibrate the hard-decision
        threshold.
    - config (overrides constructor defaults at runtime): ``mode``, ``k``,
      ``z_dim``, ``n_steps``, ``lr``, ``lambda_norm``, ``calibration_fpr``.

    Main output
    -----------
    - ``membership_scores`` (= -reconstruction_error; higher -> more likely
      member), ``membership_preds`` (error below the reference-calibrated
      threshold when available), plus per-query reconstruction errors and the
      decision threshold in ``intermediate_outputs``.
    """

    def __init__(
        self,
        mode: str = "fbb",
        k: int = 5,
        z_dim: int = 16,
        n_steps: int = 300,
        lr: float = 0.1,
        lambda_norm: float = 1e-3,
        calibration_fpr: float = 0.05,
        device: Optional[str] = None,
    ) -> None:
        if mode not in {"fbb", "pbb"}:
            raise ValueError(f"mode must be 'fbb' or 'pbb', got {mode!r}.")
        self.mode = mode
        self.k = int(k)
        self.z_dim = int(z_dim)
        self.n_steps = int(n_steps)
        self.lr = float(lr)
        self.lambda_norm = float(lambda_norm)
        self.calibration_fpr = float(calibration_fpr)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._knn: Optional[NearestNeighbors] = None
        self._threshold: Optional[float] = None

    # ------------------------------------------------------------------- fit
    def fit(self, attack_input: AttackInput) -> "GANLeaksAttack":
        cfg = attack_input.config
        self.mode = cfg.get("mode", self.mode)
        if self.mode not in {"fbb", "pbb"}:
            raise ValueError(f"mode must be 'fbb' or 'pbb', got {self.mode!r}.")
        self.k = int(cfg.get("k", self.k))
        self.z_dim = int(cfg.get("z_dim", self.z_dim))
        self.calibration_fpr = float(cfg.get("calibration_fpr", self.calibration_fpr))

        calibration_X = (attack_input.reference_data or {}).get("calibration_X")
        if self.mode == "fbb":
            self._knn = NearestNeighbors(n_neighbors=self.k, n_jobs=-1)
            self._knn.fit(_flatten(_require_generated_samples(attack_input)))
            if calibration_X is not None:
                cal_errors, _ = self._knn.kneighbors(_flatten(calibration_X), self.k)
                self._threshold = self._calibration_threshold(cal_errors.mean(axis=1))
        # pbb: nothing to fit; the threshold is computed in infer once we can
        # invert the calibration set through the generator.
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        cfg = attack_input.config
        n_steps = int(cfg.get("n_steps", self.n_steps))
        lr = float(cfg.get("lr", self.lr))
        lambda_norm = float(cfg.get("lambda_norm", self.lambda_norm))

        X = _flatten(attack_input.samples)
        if X is None:
            raise ValueError("GANLeaksAttack.infer requires attack_input.samples.")

        if self.mode == "fbb":
            errors = self._fbb_errors(X)
        else:
            if attack_input.target_model is None:
                raise ValueError("mode='pbb' requires attack_input.target_model (a generator).")
            self.z_dim = int(cfg.get("z_dim", self.z_dim))
            generator = attack_input.target_model.to(self.device).eval()
            errors = self._pbb_errors(X, generator, n_steps, lr, lambda_norm)

        scores = -errors  # higher -> more likely member

        # Calibrate threshold on known non-member reference data if available.
        threshold = self._threshold
        if threshold is None:
            calibration_X = (attack_input.reference_data or {}).get("calibration_X")
            if calibration_X is not None:
                cal_errors = self._reference_errors(_flatten(calibration_X), attack_input, n_steps, lr, lambda_norm)
                threshold = self._calibration_threshold(cal_errors)
                self._threshold = threshold

        if threshold is not None:
            preds = (errors < threshold).astype(np.int64)
        else:
            preds = (scores >= 0.0).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "reconstruction_error": errors,
                "decision_threshold": threshold,
                "mode": self.mode,
            },
            metadata={
                "attack_name": "gan_leaks",
                "mode": self.mode,
                "k": self.k if self.mode == "fbb" else None,
                "z_dim": self.z_dim if self.mode == "pbb" else None,
                "calibration_fpr": self.calibration_fpr,
            },
        )

    # -------------------------------------------------------------- internals
    def _fbb_errors(self, X: np.ndarray) -> np.ndarray:
        if self._knn is None:
            raise RuntimeError("FBB index not built; call fit() first.")
        dists, _ = self._knn.kneighbors(X, self.k)
        return dists.mean(axis=1)

    def _pbb_errors(
        self,
        X: np.ndarray,
        generator: nn.Module,
        n_steps: int,
        lr: float,
        lambda_norm: float,
    ) -> np.ndarray:
        """Invert each query by optimising a latent to minimise reconstruction error."""
        X_t = torch.from_numpy(X).to(self.device).float()
        n = X_t.shape[0]
        # Deterministic zero init (matches the reference 'zero' initialisation);
        # Adam escapes it via the learning-rate schedule.
        z = torch.zeros(n, self.z_dim, device=self.device, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            recon = generator(z)
            per_loss = ((recon - X_t) ** 2).flatten(1).sum(dim=1)
            loss = per_loss.mean() + lambda_norm * (z ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            recon = generator(z)
            errors = ((recon - X_t) ** 2).flatten(1).sum(dim=1).cpu().numpy()
        return errors

    def _reference_errors(
        self,
        cal_X: np.ndarray,
        attack_input: AttackInput,
        n_steps: int,
        lr: float,
        lambda_norm: float,
    ) -> np.ndarray:
        if self.mode == "fbb":
            return self._fbb_errors(cal_X)
        return self._pbb_errors(cal_X, attack_input.target_model.to(self.device).eval(), n_steps, lr, lambda_norm)

    def _calibration_threshold(self, cal_errors: np.ndarray) -> Optional[float]:
        """Reconstruction-error threshold calibrated for a target FPR.

        Members have *low* reconstruction error, so a query is flagged as member
        when ``error < threshold``. Setting the threshold at the ``fpr``
        quantile of known non-member errors yields FPR ≈ ``calibration_fpr``.
        """
        fpr = self.calibration_fpr
        if not (0.0 < fpr < 1.0) or len(cal_errors) == 0:
            return None
        return float(np.quantile(cal_errors, fpr))


# --------------------------------------------------------------------- helpers
def _require_generated_samples(attack_input: AttackInput) -> np.ndarray:
    gen = (attack_input.signals or {}).get("generated_samples")
    if gen is None:
        raise ValueError(
            "mode='fbb' requires signals['generated_samples'] (pre-generated "
            "samples to build the k-NN index over)."
        )
    return _flatten(gen)


def _flatten(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if arr.ndim <= 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


__all__ = ["GANLeaksAttack"]
