"""
Utilities for RMIA reference-model training and score computation.

It reimplements the reference-model
training pattern and the RMIA scoring logic in reusable pieces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReferencePredictions:
    """Collected per-sample reference predictions."""

    in_model_predictions: Dict[int, List[np.ndarray]]
    out_model_predictions: Dict[int, List[np.ndarray]]
    sample_labels: np.ndarray


def extract_true_label_confidences(probabilities: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """Extract confidence for the true class from a probability matrix."""
    return probabilities[np.arange(len(probabilities)), true_labels]


def _rowwise_nanmean(values: np.ndarray) -> np.ndarray:
    counts = np.sum(~np.isnan(values), axis=1)
    sums = np.nansum(values, axis=1)
    means = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    return means


class RMIAReferenceManager:
    """
    Train reference models and collect per-sample predictions for RMIA.

    Global sample index convention:
    - train_X samples use indices [0, len(train_X))
    - test_X samples use indices [len(train_X), len(train_X) + len(test_X))
    """

    def __init__(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        test_X: np.ndarray,
        test_y: np.ndarray,
        model_factory: Callable[[], nn.Module],
        train_fn: Optional[Callable[[nn.Module, DataLoader, torch.device, Dict[str, Any]], None]] = None,
        train_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.train_X = np.asarray(train_X)
        self.train_y = np.asarray(train_y)
        self.test_X = np.asarray(test_X)
        self.test_y = np.asarray(test_y)
        self.model_factory = model_factory
        self.train_fn = train_fn or self._default_train_fn
        self.train_config = train_config or {}
        self.batch_size = batch_size
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.total_samples = len(self.train_X) + len(self.test_X)
        self.reference_predictions = ReferencePredictions(
            in_model_predictions={},
            out_model_predictions={},
            sample_labels=np.full(self.total_samples, -1, dtype=np.int64),
        )

    def train_reference_models(
        self,
        data_sizes: List[int],
        random_seed_num: int = 5,
        reference_model_number: int = 10,
        ensure_full_coverage: bool = True,
        min_reference_observations: int = 2,
        required_indices: Optional[np.ndarray] = None,
    ) -> None:
        """Train reference models and collect in/out predictions."""
        if reference_model_number < 2:
            raise ValueError("reference_model_number must be at least 2.")
        if min_reference_observations < 1:
            raise ValueError("min_reference_observations must be at least 1.")

        for data_size in data_sizes:
            for seed_offset in range(random_seed_num):
                seed = data_size + seed_offset
                self._set_seed(seed)

                train_sample_X, train_sample_y, train_indices = self._sample_data(
                    self.train_X, self.train_y, data_size, seed
                )
                test_sample_X, test_sample_y, test_indices = self._sample_data(
                    self.test_X, self.test_y, data_size, seed
                )

                merged_X = np.concatenate([train_sample_X, test_sample_X], axis=0)
                merged_y = np.concatenate([train_sample_y, test_sample_y], axis=0)
                merged_indices = np.concatenate(
                    [train_indices, test_indices + len(self.train_X)],
                    axis=0,
                )

                self.reference_predictions.sample_labels[merged_indices] = merged_y

                for split_num in range(reference_model_number // 2):
                    split_seed = seed + split_num
                    x_a, x_b, y_a, y_b, idx_a, idx_b = train_test_split(
                        merged_X,
                        merged_y,
                        merged_indices,
                        test_size=0.5,
                        random_state=split_seed,
                    )

                    self._train_and_record(x_a, y_a, x_b, idx_a, idx_b)
                    self._train_and_record(x_b, y_b, x_a, idx_b, idx_a)

        if ensure_full_coverage:
            self._ensure_reference_coverage(
                required_indices=required_indices,
                min_reference_observations=min_reference_observations,
            )

    def _ensure_reference_coverage(
        self,
        required_indices: Optional[np.ndarray],
        min_reference_observations: int,
    ) -> None:
        """Train reciprocal full-universe pairs to close reference gaps."""
        if self.total_samples < 2:
            raise ValueError("At least two reference samples are required.")

        if required_indices is None:
            required = np.arange(self.total_samples, dtype=np.int64)
        else:
            required = np.unique(np.asarray(required_indices, dtype=np.int64))
        if required.size == 0:
            return
        if required.min() < 0 or required.max() >= self.total_samples:
            raise IndexError("required_indices contains an out-of-range sample index.")

        merged_X = np.concatenate([self.train_X, self.test_X], axis=0)
        merged_y = np.concatenate([self.train_y, self.test_y], axis=0)
        merged_indices = np.arange(self.total_samples, dtype=np.int64)
        self.reference_predictions.sample_labels[:] = merged_y

        coverage_round = 0
        while True:
            min_in = min(
                len(self.reference_predictions.in_model_predictions.get(int(i), []))
                for i in required
            )
            min_out = min(
                len(self.reference_predictions.out_model_predictions.get(int(i), []))
                for i in required
            )
            if min_in >= min_reference_observations and min_out >= min_reference_observations:
                return

            split_seed = 1_000_003 + coverage_round
            x_a, x_b, y_a, y_b, idx_a, idx_b = train_test_split(
                merged_X,
                merged_y,
                merged_indices,
                test_size=0.5,
                random_state=split_seed,
            )
            self._train_and_record(x_a, y_a, x_b, idx_a, idx_b)
            self._train_and_record(x_b, y_b, x_a, idx_b, idx_a)
            coverage_round += 1

    def get_sample_predictions(self, sample_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """Gather per-sample in/out true-label confidences."""
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        batch_size = len(sample_indices)

        all_in_preds: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
        all_out_preds: List[List[np.ndarray]] = [[] for _ in range(batch_size)]

        for i, sample_idx in enumerate(sample_indices):
            if sample_idx in self.reference_predictions.in_model_predictions:
                all_in_preds[i].extend(self.reference_predictions.in_model_predictions[sample_idx])
            if sample_idx in self.reference_predictions.out_model_predictions:
                all_out_preds[i].extend(self.reference_predictions.out_model_predictions[sample_idx])

        max_in_models = max((len(preds) for preds in all_in_preds), default=0)
        max_out_models = max((len(preds) for preds in all_out_preds), default=0)

        in_confs = np.full((batch_size, max_in_models), np.nan, dtype=np.float64)
        out_confs = np.full((batch_size, max_out_models), np.nan, dtype=np.float64)
        true_labels = self.reference_predictions.sample_labels[sample_indices]

        for i in range(batch_size):
            label = true_labels[i]
            if label < 0:
                continue
            for j, pred in enumerate(all_in_preds[i]):
                in_confs[i, j] = pred[label]
            for j, pred in enumerate(all_out_preds[i]):
                out_confs[i, j] = pred[label]

        return {
            "in_confs": in_confs,
            "out_confs": out_confs,
            "true_labels": true_labels,
        }

    def compute_rmia_scores(
        self,
        target_model_preds: np.ndarray,
        sample_indices: np.ndarray,
        population_preds: np.ndarray,
        population_indices: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        population_true_labels: Optional[np.ndarray] = None,
        a: float = 0.0,
        gamma: float = 1.0,
        return_details: bool = False,
        comparison_chunk_size: int = 1024,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute RMIA scores.

        The score follows the reference logic:
            ratio_x = p_target(x_y) / Pr(x)
            ratio_z = p_target(z_y) / Pr(z)
            score(x) = mean_z[ ratio_x / ratio_z > gamma ]

        Higher score means more likely member.
        """
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        population_indices = np.asarray(population_indices, dtype=np.int64)

        target_ref = self.get_sample_predictions(sample_indices)
        target_in_confs = target_ref["in_confs"]
        target_out_confs = target_ref["out_confs"]
        stored_target_labels = target_ref["true_labels"]

        if true_labels is None:
            true_labels = stored_target_labels.copy()
        else:
            true_labels = np.asarray(true_labels, dtype=np.int64)
        unresolved_target_mask = true_labels < 0
        true_labels[unresolved_target_mask] = stored_target_labels[unresolved_target_mask]

        target_confs = extract_true_label_confidences(np.asarray(target_model_preds), true_labels)
        target_all_ref_confs = _concat_nan_columns(target_in_confs, target_out_confs)
        pr_x_out = _rowwise_nanmean(target_all_ref_confs)
        pr_x = 0.5 * ((1.0 + a) * pr_x_out + (1.0 - a))
        ratio_x = target_confs / (pr_x + 1e-10)

        population_ref = self.get_sample_predictions(population_indices)
        population_in_confs = population_ref["in_confs"]
        population_out_confs = population_ref["out_confs"]
        stored_population_labels = population_ref["true_labels"]

        if population_true_labels is None:
            population_true_labels = stored_population_labels.copy()
        else:
            population_true_labels = np.asarray(population_true_labels, dtype=np.int64)
        unresolved_population_mask = population_true_labels < 0
        population_true_labels[unresolved_population_mask] = stored_population_labels[unresolved_population_mask]

        population_target_confs = extract_true_label_confidences(
            np.asarray(population_preds),
            population_true_labels,
        )
        population_all_ref_confs = _concat_nan_columns(population_in_confs, population_out_confs)
        pr_z = _rowwise_nanmean(population_all_ref_confs)
        ratio_z = population_target_confs / (pr_z + 1e-10)

        valid_target_mask = ~(np.isnan(pr_x_out) | (true_labels < 0))
        valid_population_mask = ~(np.isnan(pr_z) | (population_true_labels < 0))

        scores = np.zeros(len(sample_indices), dtype=np.float64)
        if np.any(valid_target_mask) and np.any(valid_population_mask):
            valid_ratio_x = ratio_x[valid_target_mask]
            valid_ratio_z = ratio_z[valid_population_mask]
            counts = _count_ratio_exceedances(
                ratio_x=valid_ratio_x,
                ratio_z=valid_ratio_z,
                gamma=gamma,
                chunk_size=comparison_chunk_size,
            )
            scores[valid_target_mask] = counts / len(valid_ratio_z)

        if return_details:
            return scores, {
                "valid_target_mask": valid_target_mask,
                "valid_population_mask": valid_population_mask,
                "target_confidences": target_confs,
                "population_confidences": population_target_confs,
                "pr_x_out": pr_x_out,
                "pr_x": pr_x,
                "pr_z": pr_z,
                "ratio_x": ratio_x,
                "ratio_z": ratio_z,
            }
        return scores


    def _train_and_record(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        eval_X: np.ndarray,
        train_indices: np.ndarray,
        eval_indices: np.ndarray,
    ) -> None:
        model = self.model_factory().to(self.device)
        train_loader = self._build_loader(train_X, train_y, shuffle=True)
        self.train_fn(model, train_loader, self.device, self.train_config)

        train_probs = self._predict_probabilities(model, train_X)
        eval_probs = self._predict_probabilities(model, eval_X)

        for i, sample_idx in enumerate(train_indices):
            self.reference_predictions.in_model_predictions.setdefault(int(sample_idx), []).append(train_probs[i])
        for i, sample_idx in enumerate(eval_indices):
            self.reference_predictions.out_model_predictions.setdefault(int(sample_idx), []).append(eval_probs[i])

    def _sample_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        size: int,
        seed: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if size > len(X):
            raise ValueError(f"Requested sample size {size} exceeds dataset size {len(X)}.")
        if size == len(X):
            return X.copy(), y.copy(), np.arange(len(X), dtype=np.int64)

        _, X_sample, _, y_sample, _, sample_indices = train_test_split(
            X,
            y,
            np.arange(len(X)),
            test_size=size,
            random_state=seed,
        )
        return X_sample, y_sample, sample_indices

    def _predict_probabilities(self, model: nn.Module, samples: np.ndarray) -> np.ndarray:
        model.eval()
        loader = self._build_loader(samples, labels=None, shuffle=False)
        all_probs: List[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                batch_x = batch[0].to(self.device)
                logits = model(batch_x)
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs.detach().cpu())
        return torch.cat(all_probs, dim=0).numpy().astype(np.float64)

    def _build_loader(
        self,
        samples: np.ndarray,
        labels: Optional[np.ndarray],
        shuffle: bool,
    ) -> DataLoader:
        x_tensor = torch.tensor(samples, dtype=torch.float32)
        if labels is None:
            dataset = TensorDataset(x_tensor)
        else:
            y_tensor = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=shuffle)

    def _set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _default_train_fn(
        model: nn.Module,
        train_loader: DataLoader,
        device: torch.device,
        train_config: Dict[str, Any],
    ) -> None:
        epochs = int(train_config.get("epochs", 20))
        lr = float(train_config.get("lr", 1e-3))
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad()
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()


def _count_ratio_exceedances(
    ratio_x: np.ndarray,
    ratio_z: np.ndarray,
    gamma: float,
    chunk_size: int,
) -> np.ndarray:
    """Count RMIA pairwise comparisons using bounded-memory 2-D blocks."""
    if chunk_size < 1:
        raise ValueError("comparison_chunk_size must be at least 1.")

    ratio_x = np.asarray(ratio_x, dtype=np.float64).reshape(-1)
    ratio_z = np.asarray(ratio_z, dtype=np.float64).reshape(-1)
    counts = np.zeros(len(ratio_x), dtype=np.int64)

    for x_start in range(0, len(ratio_x), chunk_size):
        x_stop = min(x_start + chunk_size, len(ratio_x))
        x_values = ratio_x[x_start:x_stop, None]
        block_counts = np.zeros(x_stop - x_start, dtype=np.int64)
        for z_start in range(0, len(ratio_z), chunk_size):
            z_stop = min(z_start + chunk_size, len(ratio_z))
            z_values = ratio_z[None, z_start:z_stop]
            block_counts += np.sum(
                x_values / (z_values + 1e-10) > gamma,
                axis=1,
            )
        counts[x_start:x_stop] = block_counts
    return counts


def _concat_nan_columns(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.size == 0 and right.size == 0:
        return np.full((left.shape[0], 1), np.nan, dtype=np.float64)
    if left.size == 0:
        return right
    if right.size == 0:
        return left
    return np.concatenate([left, right], axis=1)
