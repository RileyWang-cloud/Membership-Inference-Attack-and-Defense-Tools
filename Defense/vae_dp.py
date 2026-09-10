"""
DP-SGD-trained Variational Autoencoder as an MIA defense, wrapped with the
project's minimal defense interface.

Reimplemented from scratch following ``DEFENSE_INTERFACE_DESIGN_ZH.md``; the
earlier ``vae_mia_core.py`` was removed as unreliable. That file was a
TensorFlow/Keras *attack* harness (it trained a VAE then ran reconstruction /
MC-PCA / SSIM membership-inference attacks against it) with ``use_dp`` /
``noise_multiplier`` / ``l2_norm_clip`` config flags that were never wired into
the training loop — i.e. the "DP" defence existed only as dead code. The
reference project's actual contribution is using DP-SGD to train a generative
model so it resists membership inference; this module implements that defence.

Reference
---------
Amazon AWS "security-research-vae-dp-mia"; Abadi et al., "Deep Learning with
Differential Privacy" (DP-SGD, CCS 2016).

Defence idea
------------
A VAE trained with plain SGD overfits its training set, so members reconstruct
with markedly lower error than non-members — a reconstruction-based MIA exploits
exactly that gap. DP-SGD bounds the per-step sensitivity by clipping per-sample
gradients to a norm ``C`` and adding calibrated Gaussian noise (std
``sigma * C``) to their sum before averaging. The result is a VAE that fits the
data distribution without memorising individual samples, shrinking the
member/non-member reconstruction-error gap and thus lowering MIA success — at
the cost of some reconstruction quality (the privacy-utility trade-off).

This is a ``training_time`` defence whose main output is ``defended_model`` (the
trained VAE). DP-SGD is implemented manually (per-sample clipping + noise), so
no Opacus dependency is required.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from Defense.base import (
    BaseDefense,
    DefenseEvaluationResult,
    DefenseInput,
    DefenseOutput,
)


class VAE(nn.Module):
    """MLP variational autoencoder for vector data.

    ``forward`` returns ``(reconstruction, z_mean, z_log_var)``. The encoder
    outputs ``2 * latent_dim`` units (mean and log-variance concatenated).
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden: Sequence[int] = (64, 64)) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.kl_weight: float = 1.0
        enc_layers: List[nn.Module] = []
        last = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        enc_layers.append(nn.Linear(last, 2 * self.latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        last = self.latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        dec_layers.append(nn.Linear(last, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z_mean, z_log_var = h[..., : self.latent_dim], h[..., self.latent_dim :]
        return z_mean, z_log_var

    def reparameterize(self, z_mean: torch.Tensor, z_log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * z_log_var)
        return z_mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        recon = self.decode(z)
        return recon, z_mean, z_log_var

    def elbo_per_sample(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample VAE loss (reconstruction MSE + ``kl_weight`` * KL), shape ``(N,)``."""
        recon, z_mean, z_log_var = self.forward(x)
        recon_loss = ((recon - x) ** 2).flatten(1).sum(dim=1)
        kl = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp()).sum(dim=1)
        return recon_loss + self.kl_weight * kl

    @torch.no_grad()
    def reconstruction_error(self, x: torch.Tensor, n_passes: int = 5) -> torch.Tensor:
        """Mean per-sample reconstruction MSE over ``n_passes`` stochastic passes."""
        errs = []
        for _ in range(n_passes):
            recon, _, _ = self.forward(x)
            errs.append(((recon - x) ** 2).flatten(1).mean(dim=1))
        return torch.stack(errs).mean(dim=0)


class VAEDPDefense(BaseDefense):
    """
    DP-SGD-trained VAE defence against membership inference.

    defence_mode: training_time
    ---------------------------
    Trains a VAE on the member data and returns it as ``defended_model``.

    Required DefenseInput fields
    ----------------------------
    - train_data
        Member training data, shape ``(N, D)`` (numpy array or torch tensor).
    - test_data
        Non-member data used for utility/privacy evaluation.

    Optional
    --------
    - model_factory
        Callable returning a fresh ``VAE`` (or compatible). If omitted, an MLP
        VAE is built from ``defense_config``.
    - defense_config (overrides constructor defaults at runtime): ``use_dp``,
      ``noise_multiplier``, ``max_grad_norm``, ``latent_dim``, ``hidden``,
      ``kl_weight`` (β; small values expose per-sample memorisation that DP
      then mitigates), ``epochs``, ``lr``, ``batch_size``, ``recon_passes``.

    Main output
    -----------
    - ``defended_model`` (the trained VAE), plus training loss history in
      ``artifacts``. When ``eval_config`` is provided, ``evaluate`` reports
      reconstruction MSE (utility) and a reconstruction-MIA AUROC (privacy).
    """

    name: str = "vae_dp"
    defense_family: str = "dp_sgd"
    defense_mode: str = "training_time"
    supported_model_types: list[str] = ["vae", "generative"]
    required_input_keys: list[str] = ["train_data"]
    optional_input_keys: list[str] = ["test_data", "model_factory"]

    def __init__(
        self,
        latent_dim: int = 8,
        hidden: Sequence[int] = (64, 64),
        use_dp: bool = True,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        epochs: int = 30,
        lr: float = 1e-2,
        batch_size: int = 64,
        recon_passes: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.latent_dim = int(latent_dim)
        self.hidden = tuple(int(h) for h in hidden)
        self.use_dp = bool(use_dp)
        self.noise_multiplier = float(noise_multiplier)
        self.max_grad_norm = float(max_grad_norm)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.recon_passes = int(recon_passes)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._vae: Optional[VAE] = None
        self._history: Dict[str, list] = {"loss": []}

    # ------------------------------------------------------------------- fit
    def fit(self, defense_input: DefenseInput) -> "VAEDPDefense":
        if defense_input.train_data is None:
            raise ValueError("VAEDPDefense.fit requires defense_input.train_data.")
        cfg = defense_input.defense_config
        use_dp = bool(cfg.get("use_dp", self.use_dp))
        epochs = int(cfg.get("epochs", self.epochs))
        lr = float(cfg.get("lr", self.lr))
        batch_size = int(cfg.get("batch_size", self.batch_size))
        latent_dim = int(cfg.get("latent_dim", self.latent_dim))
        hidden = tuple(cfg.get("hidden", self.hidden))

        X = _to_tensor_2d(defense_input.train_data).to(self.device)
        input_dim = int(X.shape[1])

        if defense_input.model_factory is not None:
            vae = defense_input.model_factory().to(self.device)
        else:
            vae = VAE(input_dim, latent_dim, hidden).to(self.device)
        vae.kl_weight = float(cfg.get("kl_weight", 1.0))

        if use_dp:
            noise_multiplier = float(cfg.get("noise_multiplier", self.noise_multiplier))
            max_grad_norm = float(cfg.get("max_grad_norm", self.max_grad_norm))
            self._train_dp(vae, X, epochs, batch_size, lr, noise_multiplier, max_grad_norm)
        else:
            self._train_plain(vae, X, epochs, batch_size, lr)

        self._vae = vae
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        if self._vae is None:
            raise RuntimeError("VAEDPDefense must be fitted before infer().")
        return DefenseOutput(
            defended_model=self._vae,
            artifacts={"history": dict(self._history), "use_dp": self.use_dp},
            metadata={
                "defense_name": self.name,
                "defense_mode": self.defense_mode,
                "latent_dim": self.latent_dim,
            },
        )

    # ------------------------------------------------------------- evaluate
    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        """Utility (reconstruction MSE) + privacy (reconstruction-MIA AUROC).

        Lower MIA AUROC = better privacy (members are less distinguishable).
        """
        cfg = defense_input.defense_config
        n_passes = int(cfg.get("recon_passes", self.recon_passes))
        vae = defense_output.defended_model
        if vae is None:
            raise ValueError("evaluate requires defense_output.defended_model.")

        X_train = _to_tensor_2d(defense_input.train_data).to(self.device)
        X_test = _to_tensor_2d(defense_input.test_data).to(self.device) if defense_input.test_data is not None else None

        with torch.no_grad():
            train_err = vae.reconstruction_error(X_train, n_passes).cpu().numpy()
            train_mse = float(train_err.mean())
            utility_metrics = {"reconstruction_mse_train": train_mse}
            privacy_metrics: Dict[str, float] = {}
            if X_test is not None and len(X_test) > 0:
                test_err = vae.reconstruction_error(X_test, n_passes).cpu().numpy()
                utility_metrics["reconstruction_mse_test"] = float(test_err.mean())
                # Reconstruction MIA: members (train) have lower error.
                errors = np.concatenate([train_err, test_err])
                labels = np.concatenate([np.ones(len(train_err)), np.zeros(len(test_err))])
                privacy_metrics["mia_auroc"] = _safe_auroc(labels, -errors)
                privacy_metrics["mia_auroc_low_fpr"] = _tpr_at_low_fpr(labels, -errors, 0.05)

        return DefenseEvaluationResult(
            utility_metrics=utility_metrics or None,
            privacy_metrics=privacy_metrics or None,
            efficiency_metrics=None,
            extra_metrics={"history": dict(self._history)},
        )

    # -------------------------------------------------------------- internals
    def _train_plain(self, vae: VAE, X: torch.Tensor, epochs: int, batch_size: int, lr: float) -> None:
        opt = torch.optim.Adam(vae.parameters(), lr=lr)
        n = X.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n, device=X.device)
            for s in range(0, n, batch_size):
                xb = X[perm[s : s + batch_size]]
                opt.zero_grad()
                vae.elbo_per_sample(xb).mean().backward()
                opt.step()
            self._record_epoch_loss(vae, X)

    def _train_dp(
        self,
        vae: VAE,
        X: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> None:
        """DP-SGD: per-sample gradient clipping + Gaussian noise on the sum."""
        params = list(vae.parameters())
        opt = torch.optim.Adam(params, lr=lr)  # DP-Adam: clipped+noised grads with Adam
        n = X.shape[0]
        vae.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=X.device)
            for s in range(0, n, batch_size):
                xb = X[perm[s : s + batch_size]]
                self._dp_step(vae, params, xb, opt, noise_multiplier, max_grad_norm)
            self._record_epoch_loss(vae, X)

    def _dp_step(
        self,
        vae: VAE,
        params: List[nn.Parameter],
        xb: torch.Tensor,
        opt: torch.optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
    ) -> None:
        batch_size = xb.shape[0]
        clipped_sum: List[torch.Tensor] = [torch.zeros_like(p) for p in params]
        # Per-sample gradients (microbatch of 1) -> exact DP-SGD clipping.
        for j in range(batch_size):
            loss_j = vae.elbo_per_sample(xb[j : j + 1]).squeeze()
            grads_j = torch.autograd.grad(loss_j, params, retain_graph=False)
            global_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads_j))
            factor = max_grad_norm / (global_norm + 1e-12)
            factor = torch.clamp(factor, max=1.0)
            for g, acc in zip(grads_j, clipped_sum):
                acc.add_(g.detach() * factor)
        # Add noise to the sum, then average (Abadi et al.).
        for acc in clipped_sum:
            acc.add_(torch.randn_like(acc) * (noise_multiplier * max_grad_norm))
            acc.div_(batch_size)
        opt.zero_grad()
        for p, acc in zip(params, clipped_sum):
            p.grad = acc
        opt.step()

    @torch.no_grad()
    def _record_epoch_loss(self, vae: VAE, X: torch.Tensor) -> None:
        # subsample for speed when recording
        idx = torch.randperm(X.shape[0], device=X.device)[:min(512, X.shape[0])]
        loss = float(vae.elbo_per_sample(X[idx]).mean().item())
        self._history["loss"].append(loss)


# --------------------------------------------------------------------- helpers
def _to_tensor_2d(value: Any) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.float().reshape(t.shape[0], -1)


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, y_score))


def _tpr_at_low_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_threshold: float) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score)
    within = fpr <= fpr_threshold
    if not within.any():
        return 0.0
    return float(tpr[within].max())


__all__ = ["VAE", "VAEDPDefense"]
