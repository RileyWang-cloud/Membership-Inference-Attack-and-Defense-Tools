"""
Enhanced Membership Inference Attack (neural-network, feature-based), wrapped
with the project's minimal attack interface.

Reimplemented from scratch following ``ATTACK_INTERFACE_DESIGN_ZH.md``; the
earlier ``Enhanced_MIA_core_algorithms.py`` was removed as unreliable. That file
carried the "Enhanced Membership Inference Attacks" title but actually contained
reference-model attacks (LiRA / Relative / Reference / Population from Carlini
et al. "Membership Inference Attacks From First Principles"), which need dozens
of reference models, overlap with the project's existing ``lira.py`` / ``rmia.py``,
and shipped a score-direction bug (``relative_attack`` negated its predictions).
The genuine "enhanced" attack of Shejwalkar & Houmansadr is a *learned* attack
model on richer per-sample features, which is what this module implements.

Reference
---------
Shejwalkar & Houmansadr, "Enhanced Membership Inference Attacks against Machine
Learning Models" (USENIX Security 2022); also related to the ml_privacy_meter
"enhanced" online attack.

Algorithm
---------
For each sample the target classifier's logits + the true label are turned into
an enhanced feature vector::

    loss         cross-entropy -log p[y]
    correctness  1{argmax z == y}
    confidence   p[y]                  (true-class probability)
    max_prob     max_c p[c]
    entropy      -sum_c p[c] log p[c]
    margin       z[y] - max_{y'!=y} z[y']
    logit_y      z[y]
    (+ one-hot label when use_label=True)

A small MLP attack classifier is trained (binary cross-entropy) to separate
shadow members from shadow non-members in this feature space, then applied to
the target queries. The membership score is the MLP's member probability, so
``higher score -> more likely member`` (design doc §7.1, §10.3).

No shadow models are trained: the attack model learns directly from the target
model's outputs on known member / non-member data (the standard "enhanced
offline" setup).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Attack.base import AttackInput, AttackOutput, BaseAttack


# Features produced by ``_extract_features`` (in this fixed order).
DEFAULT_FEATURES: Tuple[str, ...] = (
    "loss", "correctness", "confidence", "max_prob", "entropy", "margin", "logit_y",
)


class EnhancedMIAAttack(BaseAttack):
    """
    Neural-network membership inference with enhanced per-sample features.

    Required AttackInput fields
    ---------------------------
    - target_model
        A classifier returning logits of shape ``(N, C)`` (frozen during the
        attack; gradients are not needed).
    - samples
        Query samples ``X`` to attack, aligned with ``labels``.
    - labels
        Task labels ``y`` for the queries.
    - shadow_data["member_X"], shadow_data["member_y"]
        Known target-training (member) data used to fit the attack model.
    - shadow_data["nonmember_X"], shadow_data["nonmember_y"]
        Known held-out (non-member) data used to fit the attack model.

    Optional
    --------
    - signals["logits"]
        Precomputed query logits to avoid re-running the target model.
    - config (overrides constructor defaults at runtime): ``features``,
      ``use_label``, ``hidden_dims``, ``epochs``, ``lr``, ``batch_size``.

    Main output
    -----------
    - ``membership_scores`` (MLP member probability; higher -> more likely
      member), ``membership_preds`` (member prob above 0.5), plus the per-query
      feature matrix in ``intermediate_outputs``.
    """

    def __init__(
        self,
        features: Sequence[str] = DEFAULT_FEATURES,
        use_label: bool = False,
        hidden_dims: Sequence[int] = (64, 32),
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        unknown = [f for f in features if f not in DEFAULT_FEATURES]
        if unknown:
            raise ValueError(f"Unknown feature(s) {unknown}; allowed: {DEFAULT_FEATURES}")
        self.features = tuple(features)
        self.use_label = bool(use_label)
        self.hidden_dims = tuple(int(d) for d in hidden_dims)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._attack_net: Optional[nn.Module] = None
        self._feature_dim: Optional[int] = None
        self._num_classes: Optional[int] = None

    # ------------------------------------------------------------------- fit
    def fit(self, attack_input: AttackInput) -> "EnhancedMIAAttack":
        if attack_input.target_model is None:
            raise ValueError("EnhancedMIAAttack.fit requires attack_input.target_model.")
        cfg = attack_input.config
        self.features = tuple(cfg.get("features", self.features))
        self.use_label = bool(cfg.get("use_label", self.use_label))
        hidden_dims = tuple(int(d) for d in cfg.get("hidden_dims", self.hidden_dims))
        epochs = int(cfg.get("epochs", self.epochs))
        lr = float(cfg.get("lr", self.lr))
        batch_size = int(cfg.get("batch_size", self.batch_size))

        member_X, member_y, nonmember_X, nonmember_y = _require_shadow_data(attack_input)
        target_model = _freeze(attack_input.target_model).to(self.device)

        mem_logits = _logits(target_model, member_X, batch_size, self.device)
        non_logits = _logits(target_model, nonmember_X, batch_size, self.device)
        num_classes = int(max(mem_logits.shape[1], non_logits.shape[1]))
        self._num_classes = num_classes

        mem_feat = _extract_features(mem_logits, member_y, num_classes, self.features, self.use_label)
        non_feat = _extract_features(non_logits, nonmember_y, num_classes, self.features, self.use_label)
        self._feature_dim = int(mem_feat.shape[1])

        X = torch.cat([mem_feat, non_feat], dim=0)
        y = torch.cat([torch.ones(len(mem_feat)), torch.zeros(len(non_feat))]).unsqueeze(1).to(self.device)

        net = _AttackMLP(self._feature_dim, hidden_dims).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        n = X.shape[0]
        net.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=X.device)
            for s in range(0, n, batch_size):
                idx = perm[s : s + batch_size]
                prob = net(X[idx])
                loss = loss_fn(prob, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()

        self._attack_net = net
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self._attack_net is None:
            raise RuntimeError("EnhancedMIAAttack must be fitted before infer().")
        if attack_input.target_model is None and (attack_input.signals or {}).get("logits") is None:
            raise ValueError("EnhancedMIAAttack.infer requires target_model or signals['logits'].")
        if attack_input.labels is None:
            raise ValueError("EnhancedMIAAttack.infer requires attack_input.labels (task labels).")
        if attack_input.samples is None and (attack_input.signals or {}).get("logits") is None:
            raise ValueError("EnhancedMIAAttack.infer requires attack_input.samples.")

        precomputed = (attack_input.signals or {}).get("logits")
        if precomputed is not None:
            logits = _to_tensor_2d(precomputed).to(self.device)
        else:
            target_model = _freeze(attack_input.target_model).to(self.device)
            logits = _logits(target_model, attack_input.samples, self.batch_size, self.device)
        num_classes = self._num_classes or int(logits.shape[1])

        feat = _extract_features(logits, attack_input.labels, num_classes, self.features, self.use_label)
        self._attack_net.eval()
        with torch.no_grad():
            scores = self._attack_net(feat).squeeze(1).cpu().numpy()
        preds = (scores >= 0.5).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "features": feat.cpu().numpy(),
                "feature_names": list(self.features) + (["label_onehot"] if self.use_label else []),
            },
            metadata={
                "attack_name": "enhanced_mia",
                "features": list(self.features),
                "use_label": self.use_label,
                "feature_dim": int(feat.shape[1]),
            },
        )


# --------------------------------------------------------------------- helpers
class _AttackMLP(nn.Module):
    """MLP with sigmoid output producing a member probability."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers += [nn.Linear(last, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _extract_features(
    logits: torch.Tensor,
    labels: Any,
    num_classes: int,
    features: Sequence[str],
    use_label: bool,
) -> torch.Tensor:
    """Build the enhanced feature matrix from logits and labels. Shape ``(N, F)``."""
    labels_t = _to_tensor_1d(labels, torch.long).to(logits.device)
    n = logits.shape[0]
    idx = torch.arange(n, device=logits.device)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    pred = logits.argmax(dim=1)

    available = {
        "loss": -log_probs[idx, labels_t],
        "correctness": (pred == labels_t).float(),
        "confidence": probs[idx, labels_t],
        "max_prob": probs.max(dim=1).values,
        "entropy": -(probs * log_probs).sum(dim=1),
        "margin": logits[idx, labels_t] - logits.masked_fill(
            F.one_hot(labels_t, num_classes).bool(), float("-inf")
        ).max(dim=1).values,
        "logit_y": logits[idx, labels_t],
    }
    cols = [available[name].unsqueeze(1) for name in features]
    if use_label:
        cols.append(F.one_hot(labels_t, num_classes).float())
    return torch.cat(cols, dim=1)


def _require_shadow_data(attack_input: AttackInput) -> Tuple[Any, Any, Any, Any]:
    shadow = attack_input.shadow_data or {}
    missing = [k for k in ("member_X", "member_y", "nonmember_X", "nonmember_y") if k not in shadow]
    if missing:
        raise ValueError(
            "EnhancedMIAAttack.fit requires shadow_data with keys "
            "member_X / member_y / nonmember_X / nonmember_y. Missing: " + ", ".join(missing)
        )
    return shadow["member_X"], shadow["member_y"], shadow["nonmember_X"], shadow["nonmember_y"]


@torch.no_grad()
def _logits(model: nn.Module, X: Any, batch_size: int, device: torch.device) -> torch.Tensor:
    model.eval()
    Xt = _to_tensor_2d(X).to(device)
    chunks = [model(Xt[s : s + batch_size]) for s in range(0, len(Xt), batch_size)]
    return torch.cat(chunks, dim=0)


def _freeze(model: nn.Module) -> nn.Module:
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _to_tensor_2d(value: Any) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    return t.float().reshape(t.shape[0], -1)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    t = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    t = t.reshape(-1)
    return t.to(dtype) if dtype is not None else t


__all__ = ["EnhancedMIAAttack", "DEFAULT_FEATURES"]
