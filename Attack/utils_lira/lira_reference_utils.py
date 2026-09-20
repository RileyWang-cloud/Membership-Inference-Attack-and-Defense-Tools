"""
Utilities for LiRA reference-model training and score computation.

 It reimplements the necessary logic in
smaller reusable pieces so the LiRA attack can be wrapped by the project's
minimal attack interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReferencePredictions:
    """Collected per-sample reference predictions for LiRA."""

    in_model_predictions: Dict[int, List[np.ndarray]]
    out_model_predictions: Dict[int, List[np.ndarray]]
    sample_labels: np.ndarray


def logit_transform(probabilities: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Convert probabilities to logits with clipping for numerical stability."""
    # Reference caches are commonly float32.  In that dtype ``1 - 1e-10``
    # rounds back to exactly 1, which makes the subsequent logit infinite.
    # Promote before clipping so both tails have a representable safety margin.
    values = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(clipped / (1.0 - clipped))


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


def _rowwise_nanvar(values: np.ndarray, means: np.ndarray) -> np.ndarray:
    counts = np.sum(~np.isnan(values), axis=1)
    centered = values - means[:, None]
    centered[np.isnan(values)] = np.nan
    sq_sums = np.nansum(centered ** 2, axis=1)
    variances = np.full(values.shape[0], np.nan, dtype=np.float64)
    valid = counts > 0
    variances[valid] = sq_sums[valid] / counts[valid]
    return variances


class LiRAReferenceManager:
    """
    Train reference models and collect per-sample in/out confidences.

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
        """
        Train reference models and collect in/out predictions.

        The logic mirrors the reference file:
        1. Sample `data_size` points from train and test pools respectively.
        2. Merge them into one candidate set.
        3. Repeatedly split the candidate set into two halves.
        4. Train one model on each half and record predictions:
           - samples seen during training -> in_model_predictions
           - held-out samples -> out_model_predictions
        """
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
                    (
                        x_a,
                        x_b,
                        y_a,
                        y_b,
                        idx_a,
                        idx_b,
                    ) = train_test_split(
                        merged_X,
                        merged_y,
                        merged_indices,
                        test_size=0.5,
                        random_state=split_seed,
                    )

                    self._train_and_record(
                        train_X=x_a,
                        train_y=y_a,
                        eval_X=x_b,
                        eval_y=y_b,
                        train_indices=idx_a,
                        eval_indices=idx_b,
                    )
                    self._train_and_record(
                        train_X=x_b,
                        train_y=y_b,
                        eval_X=x_a,
                        eval_y=y_a,
                        train_indices=idx_b,
                        eval_indices=idx_a,
                    )

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
        """Add reciprocal full-universe models until every required sample has
        enough in/out observations.

        A reciprocal pair trains on complementary halves of the candidate
        universe. Consequently every sample receives one in-model and one
        out-model prediction per round, which deterministically closes the
        coverage holes left by sampled ``data_sizes`` configurations.
        """
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
            self._train_and_record(x_a, y_a, x_b, y_b, idx_a, idx_b)
            self._train_and_record(x_b, y_b, x_a, y_a, idx_b, idx_a)
            coverage_round += 1

    def get_sample_predictions(self, sample_indices: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Gather per-sample in/out true-label confidences.

        Returns:
            {
                "in_confs": (n_samples, max_in_models),
                "out_confs": (n_samples, max_out_models),
                "true_labels": (n_samples,)
            }
        """
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

        for i, sample_idx in enumerate(sample_indices):
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

    def compute_lira_scores(
        self,
        target_model_preds: np.ndarray,
        sample_indices: np.ndarray,
        true_labels: Optional[np.ndarray] = None,
        return_details: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compute LiRA log-likelihood-ratio scores.

        Score direction is unified with the project convention:
            higher score -> more likely member
        """
        sample_indices = np.asarray(sample_indices, dtype=np.int64)
        ref_data = self.get_sample_predictions(sample_indices)
        in_confs = ref_data["in_confs"]
        out_confs = ref_data["out_confs"]
        stored_labels = ref_data["true_labels"]
        if true_labels is None:
            true_labels = stored_labels.copy()
        else:
            true_labels = np.asarray(true_labels, dtype=np.int64)
            if true_labels.shape[0] != sample_indices.shape[0]:
                raise ValueError("true_labels must have the same length as sample_indices.")

        unresolved_label_mask = true_labels < 0
        true_labels[unresolved_label_mask] = stored_labels[unresolved_label_mask]

        target_confs = extract_true_label_confidences(
            np.asarray(target_model_preds),
            true_labels,
        )

        target_logit = logit_transform(target_confs)
        in_logit = logit_transform(in_confs)
        out_logit = logit_transform(out_confs)

        mu_in = _rowwise_nanmean(in_logit)
        mu_out = _rowwise_nanmean(out_logit)
        sigma_in = np.sqrt(_rowwise_nanvar(in_logit, mu_in))
        sigma_out = np.sqrt(_rowwise_nanvar(out_logit, mu_out))

        valid_mask = ~(np.isnan(mu_in) | np.isnan(mu_out) | (true_labels < 0))
        epsilon = 1e-10
        sigma_in = np.maximum(sigma_in, epsilon)
        sigma_out = np.maximum(sigma_out, epsilon)

        log_p_in = norm.logpdf(target_logit, loc=mu_in, scale=sigma_in)
        log_p_out = norm.logpdf(target_logit, loc=mu_out, scale=sigma_out)

        scores = np.zeros(len(sample_indices), dtype=np.float64)
        scores[valid_mask] = log_p_in[valid_mask] - log_p_out[valid_mask]
        if return_details:
            return scores, {
                "valid_mask": valid_mask,
                "target_confidences": target_confs,
                "mu_in": mu_in,
                "mu_out": mu_out,
                "sigma_in": sigma_in,
                "sigma_out": sigma_out,
            }
        return scores

    def _train_and_record(
        self,
        train_X: np.ndarray,
        train_y: np.ndarray,
        eval_X: np.ndarray,
        eval_y: np.ndarray,
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

        _ = eval_y  # kept for symmetry and possible future debugging

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
        return DataLoader(
            dataset,
            batch_size=min(self.batch_size, len(dataset)),
            shuffle=shuffle,
        )

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
