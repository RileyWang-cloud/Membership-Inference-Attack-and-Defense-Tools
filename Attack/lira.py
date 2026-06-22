"""
LiRA attack wrapped with the project's minimal attack interface.

This implementation follows the core logic from the reference-model workflow:
1. Train multiple reference models on sampled train/test subsets.
2. Collect per-sample true-label confidences for in-model and out-model cases.
3. Fit Gaussian statistics in logit space.
4. Score target samples with a log-likelihood ratio:
       score = log p_in(x) - log p_out(x)

Unified convention:
    higher membership score -> more likely member
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Attack.base import AttackInput, AttackOutput, BaseAttack
from Attack.utils_lira.lira_reference_utils import LiRAReferenceManager


class LiRAAttack(BaseAttack):
    """
    LiRA attack for classification models.

    Required reference_data fields
    ------------------------------
    Either:
    - "reference_manager": a prebuilt LiRAReferenceManager

    Or:
    - "train_X", "train_y", "test_X", "test_y"
    - "model_factory": callable returning a fresh classification model

    Required attack-time metadata
    -----------------------------
    - metadata["sample_indices"]:
        global sample indices aligned with the reference-manager convention
    """

    def __init__(
        self,
        data_sizes: Optional[list[int]] = None,
        random_seed_num: int = 5,
        reference_model_number: int = 10,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.data_sizes = data_sizes or [128]
        self.random_seed_num = random_seed_num
        self.reference_model_number = reference_model_number
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.reference_manager: Optional[LiRAReferenceManager] = None

    def fit(self, attack_input: AttackInput) -> "LiRAAttack":
        reference_data = attack_input.reference_data
        if reference_data is None:
            raise ValueError("reference_data is required for LiRAAttack.fit().")

        if "reference_manager" in reference_data:
            self.reference_manager = reference_data["reference_manager"]
            return self

        required_keys = ["train_X", "train_y", "test_X", "test_y", "model_factory"]
        missing_keys = [key for key in required_keys if key not in reference_data]
        if missing_keys:
            raise ValueError(
                "reference_data must provide either 'reference_manager' or all of "
                f"{required_keys}. Missing: {missing_keys}"
            )

        manager = LiRAReferenceManager(
            train_X=np.asarray(reference_data["train_X"]),
            train_y=np.asarray(reference_data["train_y"]),
            test_X=np.asarray(reference_data["test_X"]),
            test_y=np.asarray(reference_data["test_y"]),
            model_factory=reference_data["model_factory"],
            train_fn=reference_data.get("train_fn"),
            train_config=reference_data.get("train_config", {}),
            batch_size=int(reference_data.get("batch_size", self.batch_size)),
            device=str(self.device),
        )
        manager.train_reference_models(
            data_sizes=reference_data.get("data_sizes", self.data_sizes),
            random_seed_num=int(reference_data.get("random_seed_num", self.random_seed_num)),
            reference_model_number=int(
                reference_data.get("reference_model_number", self.reference_model_number)
            ),
        )
        self.reference_manager = manager
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.reference_manager is None:
            raise RuntimeError("LiRAAttack must be fitted before infer().")

        sample_indices = self._resolve_sample_indices(attack_input)
        target_probs = self._get_target_probabilities(attack_input)
        true_labels = _to_numpy_1d(attack_input.labels).astype(np.int64) if attack_input.labels is not None else None
        scores, details = self.reference_manager.compute_lira_scores(
            target_probs,
            sample_indices,
            true_labels=true_labels,
            return_details=True,
        )
        preds = (scores >= 0.0).astype(np.int64)

        return AttackOutput(
            membership_scores=scores,
            membership_preds=preds,
            intermediate_outputs={
                "target_probabilities": target_probs,
                "sample_indices": sample_indices,
                "valid_reference_mask": details["valid_mask"],
                "reference_coverage": float(np.mean(details["valid_mask"])) if len(details["valid_mask"]) > 0 else 0.0,
                "target_confidences": details["target_confidences"],
                "mu_in": details["mu_in"],
                "mu_out": details["mu_out"],
            },
            metadata={
                "attack_name": "lira",
                "data_sizes": self.data_sizes,
                "random_seed_num": self.random_seed_num,
                "reference_model_number": self.reference_model_number,
            },
        )

    def _resolve_sample_indices(self, attack_input: AttackInput) -> np.ndarray:
        if "sample_indices" in attack_input.metadata:
            return np.asarray(attack_input.metadata["sample_indices"], dtype=np.int64)
        if attack_input.reference_data is not None and "sample_indices" in attack_input.reference_data:
            return np.asarray(attack_input.reference_data["sample_indices"], dtype=np.int64)
        raise ValueError(
            "LiRAAttack requires global sample indices in attack_input.metadata['sample_indices'] "
            "or attack_input.reference_data['sample_indices']."
        )

    def _get_target_probabilities(self, attack_input: AttackInput) -> np.ndarray:
        if attack_input.signals is not None and "probabilities" in attack_input.signals:
            return _to_numpy_2d(attack_input.signals["probabilities"])

        if attack_input.target_model is None:
            raise ValueError(
                "target_model is required when signals['probabilities'] is not provided."
            )

        return _predict_probabilities(
            model=attack_input.target_model,
            samples=attack_input.samples,
            batch_size=int(attack_input.config.get("batch_size", self.batch_size)),
            device=self.device,
        )


def _predict_probabilities(
    model: nn.Module,
    samples: Any,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = _build_loader(samples, labels=None, batch_size=batch_size, shuffle=False)
    model.eval()
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch_x = batch[0].to(device)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu())

    return torch.cat(all_probs, dim=0).numpy().astype(np.float64)


def _build_loader(samples: Any, labels: Optional[Any], batch_size: int, shuffle: bool) -> DataLoader:
    x_tensor = _to_tensor_any(samples)
    if labels is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_tensor = _to_tensor_1d(labels, dtype=torch.long)
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _to_tensor_any(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value, dtype=torch.float32)


def _to_tensor_1d(value: Any, dtype: Optional[torch.dtype] = None) -> Optional[torch.Tensor]:
    if value is None:
        return None
    tensor = value.detach().cpu() if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    tensor = tensor.reshape(-1)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _to_numpy_1d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().reshape(-1)
    return np.asarray(value).reshape(-1)


def _to_numpy_2d(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


__all__ = ["LiRAAttack"]
