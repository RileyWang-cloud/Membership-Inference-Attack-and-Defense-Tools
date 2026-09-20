"""
EncoderMI: membership inference against contrastive-learning encoders.

This module adapts the EncoderMI workflow implemented by
`Attack/encodermi.py` + `Attack/utils_encodermi/` (Liu et al., "Membership
Inference via Contrastive Learning") behind the project's minimal attack
interface (`Attack.base`):
    AttackInput -> AttackOutput

Attack idea (vector-based variant, `utils_MI_V.py`)
----------------------------------------------------
For every sample, extract the encoder feature under N random augmentations,
L2-normalize them and take the sorted pairwise cosine similarities. Members
are reproduced more consistently by the encoder, so their sorted similarity
vector differs from non-members'. A small binary attack model trained on the
shadow encoder's member/non-member similarity vectors scores the target
encoder's samples.

Unified convention:
    higher membership score -> more likely member
(attack-model probabilities already satisfy this direction).

The original CIFAR-10/MoCo training pipeline stays in `encodermi.py`;
this wrapper consumes already-trained encoders and datasets.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from Attack.base import AttackInput, AttackOutput, BaseAttack


class AttackMLP(nn.Module):
    """Binary attack classifier over sorted cosine-similarity vectors."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class EncoderMIAttack(BaseAttack):
    """
    Vector-based EncoderMI: augmentation-consistency membership inference.

    Required fields
    ---------------
    - attack_input.target_model:
        trained target encoder (nn.Module mapping samples to feature
        vectors), unless signals["features"] is given at inference time
    - attack_input.samples:
        target samples to attack

    fit() inputs (any one of)
    -------------------------
    Format A: precomputed sorted-cosine-similarity features
        shadow_data = {
            "member_features": ndarray (n_m, n_pairs),
            "nonmember_features": ndarray (n_nm, n_pairs),
        }
    Format B: shadow encoder + raw data (features computed here)
        shadow_data = {
            "shadow_model": nn.Module encoder,
            "member_samples": raw member samples of the shadow encoder,
            "nonmember_samples": raw non-member samples,
        }

    Useful config keys
    ------------------
    - n_augmentations (int, default 10): number of augmented views
    - augment_fn: callable x -> augmented x; defaults to
      torchvision RandomResizedCrop for 4-D image batches and Gaussian
      noise otherwise
    - hidden_dim (int, default 64), epochs (int, default 50),
      lr (float, default 1e-3), batch_size (int, default 128)
    """

    def __init__(
        self,
        n_augmentations: int = 10,
        hidden_dim: int = 64,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.n_augmentations = n_augmentations
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.attack_model: Optional[AttackMLP] = None
        self.is_fitted = False

    def fit(self, attack_input: AttackInput) -> "EncoderMIAttack":
        shadow_data = attack_input.shadow_data
        if shadow_data is None:
            raise ValueError("shadow_data is required for EncoderMIAttack.fit().")

        config = attack_input.config
        augment_fn = config.get("augment_fn") or _default_augment_fn()

        if shadow_data.get("member_features") is not None:
            member_features = _to_numpy_2d(shadow_data["member_features"])
            nonmember_features = _to_numpy_2d(shadow_data["nonmember_features"])
        elif shadow_data.get("shadow_model") is not None:
            encoder = shadow_data["shadow_model"].to(self.device)
            member_features = self._extract_consistency_features(
                encoder, shadow_data["member_samples"], augment_fn, config
            )
            nonmember_features = self._extract_consistency_features(
                encoder, shadow_data["nonmember_samples"], augment_fn, config
            )
        else:
            raise ValueError(
                "shadow_data must contain either precomputed *_features or "
                "'shadow_model' with member/nonmember samples."
            )

        if member_features.shape[1] != nonmember_features.shape[1]:
            raise ValueError("member/nonmember feature dimensions differ.")

        self.attack_model = self._train_attack_model(
            member_features, nonmember_features
        )
        self.is_fitted = True
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if not self.is_fitted:
            raise RuntimeError("EncoderMIAttack must be fitted before infer().")

        if attack_input.signals is not None and attack_input.signals.get(
            "features"
        ) is not None:
            features = _to_numpy_2d(attack_input.signals["features"])
        else:
            if attack_input.target_model is None:
                raise ValueError(
                    "target_model is required when signals['features'] is not provided."
                )
            augment_fn = attack_input.config.get("augment_fn") or _default_augment_fn()
            encoder = attack_input.target_model.to(self.device)
            features = self._extract_consistency_features(
                encoder, attack_input.samples, augment_fn, attack_input.config
            )

        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        self.attack_model.eval() # type: ignore[union-attr]
        with torch.no_grad():
            logits = self.attack_model(feature_tensor)
            scores = torch.sigmoid(logits).detach().cpu().numpy()

        preds = (scores >= 0.5).astype(np.int64)
        return AttackOutput(
            membership_scores=scores.astype(np.float32),
            membership_preds=preds,
            intermediate_outputs={"consistency_features": features},
            metadata={
                "attack_name": "encodermi_vector",
                "n_augmentations": int(
                    attack_input.config.get("n_augmentations", self.n_augmentations)
                ),
                "feature_dim": int(features.shape[1]),
            },
        )

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _extract_consistency_features(
        self,
        encoder: nn.Module,
        samples: Any,
        augment_fn: Callable[[torch.Tensor], torch.Tensor],
        config: Dict[str, Any],
    ) -> np.ndarray:
        """Sorted pairwise cosine similarities across augmented views."""
        n_views = int(config.get("n_augmentations", self.n_augmentations))
        batch_size = int(config.get("batch_size", self.batch_size))

        sample_tensor = _to_tensor_any(samples)
        encoder.eval()

        features_per_view: List[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_views):
                batched: List[np.ndarray] = []
                for start in range(0, sample_tensor.shape[0], batch_size):
                    batch = sample_tensor[start:start + batch_size].to(self.device)
                    augmented = augment_fn(batch)
                    feature = encoder(augmented)
                    feature = F.normalize(feature, dim=1)
                    batched.append(feature.detach().cpu().numpy())
                features_per_view.append(np.concatenate(batched, axis=0))

        # (n_views, n_samples, dim) -> per-sample sorted pairwise similarities
        views = np.stack(features_per_view, axis=0)
        n_samples = views.shape[1]
        n_pairs = n_views * (n_views - 1) // 2

        features = np.zeros((n_samples, n_pairs), dtype=np.float32)
        pair_idx = 0
        for a, b in combinations(range(n_views), 2):
            cos = np.sum(views[a] * views[b], axis=1) # already normalized
            features[:, pair_idx] = cos
            pair_idx += 1
        features.sort(axis=1) # sorted similarity vector, as in utils_MI_V
        return features

    def _train_attack_model(
        self,
        member_features: np.ndarray,
        nonmember_features: np.ndarray,
    ) -> AttackMLP:
        x = np.concatenate([member_features, nonmember_features], axis=0)
        y = np.concatenate(
            [
                np.ones(member_features.shape[0], dtype=np.float32),
                np.zeros(nonmember_features.shape[0], dtype=np.float32),
            ]
        )
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model = AttackMLP(input_dim=x.shape[1], hidden_dim=self.hidden_dim).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
        return model


def _default_augment_fn() -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Default augmentation.

    For 4-D (B, C, H, W) image batches: torchvision RandomResizedCrop,
    mirroring the original EncoderMI pipeline. Otherwise: Gaussian noise.
    """
    try:
        from torchvision.transforms import RandomResizedCrop

        crop_cache: Dict[tuple, RandomResizedCrop] = {}

        def augment_images(x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                return x
            size = (int(x.shape[-2]), int(x.shape[-1]))
            if size not in crop_cache:
                crop_cache[size] = RandomResizedCrop(size=size, scale=(0.2, 1.0))
            return crop_cache[size](x)

        return augment_images
    except Exception: # torchvision unavailable / non-image data
        def augment_noise(x: torch.Tensor) -> torch.Tensor:
            return x + 0.05 * torch.randn_like(x)

        return augment_noise


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _to_numpy_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


__all__ = ["EncoderMIAttack", "AttackMLP"]
