"""
COMPARE adapter compatible with the project's AttackInput/AttackOutput interface.

COMPARE is a recommendation-system membership inference attack. It trains an
attack classifier on shadow users and applies it to target users.

Default COMPARE features:
1. target_ndcg_10
2. target_ndcg_10 - surrogate_ndcg_10 for alpha in 0.0, 0.1, ..., 1.0

This gives the paper-style 12-dimensional feature vector. The adapter accepts
either precomputed feature matrices or raw metric rows from which those
features can be built.

Accepted training payloads in attack_input.shadow_data
------------------------------------------------------
1. Precomputed partitioned features:
   {
       "member_features": ndarray,
       "nonmember_features": ndarray,
       "feature_names": [...]
   }

2. Precomputed combined features:
   {
       "features": ndarray,
       "membership_labels": ndarray,
       "feature_names": [...]
   }

3. Raw COMPARE metric rows:
   {
       "member_metrics": [...],
       "nonmember_metrics": [...],
       "alpha_values": [0.0, ..., 1.0],
   }

Inference supports the same formats through attack_input.samples or
attack_input.signals. For evaluation, pass attack_input.membership_labels, or
include membership_labels in the inference payload.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from Attack.base import AttackInput, AttackOutput, BaseAttack


class CompareMIAAttack(BaseAttack):
    """Unified adapter for the COMPARE recommender-system MIA method."""

    def __init__(
        self,
        classifier: str = "logistic_regression",
        alpha_values: Optional[Sequence[float]] = None,
        threshold: float = 0.5,
        random_state: int = 42,
    ) -> None:
        self.classifier = classifier
        self.alpha_values = [round(0.1 * i, 1) for i in range(11)] if alpha_values is None else list(alpha_values)
        self.threshold = threshold
        self.random_state = random_state

        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        self.feature_names: Optional[List[str]] = None
        self.train_metadata: Dict[str, Any] = {}

    def fit(self, attack_input: AttackInput) -> "CompareMIAAttack":
        if attack_input.shadow_data is None:
            raise ValueError("shadow_data is required for CompareMIAAttack.fit().")

        x_train, y_train, feature_names = self._resolve_payload(
            attack_input.shadow_data,
            labels=None,
            context="shadow_data",
            require_labels=True,
        )
        self.feature_names = feature_names
        self.scaler = StandardScaler()
        x_scaled = self.scaler.fit_transform(x_train)
        self.model = self._make_classifier()
        self.model.fit(x_scaled, y_train)
        self.train_metadata = {
            "num_train_samples": int(x_train.shape[0]),
            "num_features": int(x_train.shape[1]),
            "num_members": int(np.sum(y_train == 1)),
            "num_nonmembers": int(np.sum(y_train == 0)),
        }
        return self

    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.model is None or self.scaler is None:
            raise RuntimeError("CompareMIAAttack must be fitted before infer().")

        payload = self._select_inference_payload(attack_input)
        labels = attack_input.membership_labels
        x_query, payload_labels, feature_names = self._resolve_payload(
            payload,
            labels=labels,
            context="query payload",
            require_labels=False,
        )
        if self.feature_names is not None and len(feature_names) != len(self.feature_names):
            raise ValueError(
                f"COMPARE query feature dimension mismatch: expected {len(self.feature_names)}, "
                f"got {len(feature_names)}."
            )

        x_scaled = self.scaler.transform(x_query)
        if hasattr(self.model, "predict_proba"):
            scores = self.model.predict_proba(x_scaled)[:, 1]
        else:
            scores = self.model.decision_function(x_scaled)
            scores = 1.0 / (1.0 + np.exp(-scores))
        preds = (scores >= self.threshold).astype(np.int64)

        metadata = {
            "attack_name": "compare",
            "method": "COMPARE",
            "domain": "recommender_system",
            "feature_definition": "target_ndcg_10 plus target-surrogate NDCG@10 discrepancies",
            "feature_names": feature_names,
            "alpha_values": self.alpha_values,
            "classifier": self.classifier,
            "threshold": self.threshold,
            **self.train_metadata,
        }
        intermediate_outputs = {
            "features": x_query,
            "payload_membership_labels": payload_labels,
        }
        return AttackOutput(
            membership_scores=scores.astype(np.float64),
            membership_preds=preds,
            intermediate_outputs=intermediate_outputs,
            metadata=metadata,
        )

    def _make_classifier(self) -> Any:
        if self.classifier == "logistic_regression":
            return LogisticRegression(random_state=self.random_state, max_iter=1000)
        if self.classifier == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
            )
        raise ValueError(
            "Unsupported COMPARE classifier. Use 'logistic_regression' or 'random_forest'."
        )

    def _select_inference_payload(self, attack_input: AttackInput) -> Any:
        if attack_input.signals is not None:
            if self._looks_like_payload(attack_input.signals):
                return attack_input.signals
        if self._looks_like_payload(attack_input.samples):
            return attack_input.samples
        if attack_input.samples is not None:
            return {"features": attack_input.samples}
        raise ValueError(
            "CompareMIAAttack requires query features or COMPARE metric rows in "
            "attack_input.samples or attack_input.signals."
        )

    def _looks_like_payload(self, value: Any) -> bool:
        return isinstance(value, Mapping)

    def _resolve_payload(
        self,
        payload: Any,
        labels: Optional[Any],
        context: str,
        require_labels: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        if not isinstance(payload, Mapping):
            x = self._to_numpy_2d(payload)
            y = self._coerce_labels(labels, x.shape[0], context, require_labels)
            return x, y, self._default_feature_names(x.shape[1])

        if "member_features" in payload and "nonmember_features" in payload:
            member_x = self._to_numpy_2d(payload["member_features"])
            nonmember_x = self._to_numpy_2d(payload["nonmember_features"])
            x = np.concatenate([member_x, nonmember_x], axis=0)
            y = np.concatenate(
                [
                    np.ones(member_x.shape[0], dtype=np.int64),
                    np.zeros(nonmember_x.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return x, y, self._feature_names(payload, x.shape[1])

        if "features" in payload:
            x = self._to_numpy_2d(payload["features"])
            payload_labels = payload.get("membership_labels", labels)
            y = self._coerce_labels(payload_labels, x.shape[0], context, require_labels)
            return x, y, self._feature_names(payload, x.shape[1])

        if "member_metrics" in payload and "nonmember_metrics" in payload:
            member_x, names = self._features_from_metric_rows(payload["member_metrics"], payload)
            nonmember_x, _ = self._features_from_metric_rows(payload["nonmember_metrics"], payload)
            x = np.concatenate([member_x, nonmember_x], axis=0)
            y = np.concatenate(
                [
                    np.ones(member_x.shape[0], dtype=np.int64),
                    np.zeros(nonmember_x.shape[0], dtype=np.int64),
                ],
                axis=0,
            )
            return x, y, names

        if "metrics" in payload:
            x, names = self._features_from_metric_rows(payload["metrics"], payload)
            payload_labels = payload.get("membership_labels", labels)
            y = self._coerce_labels(payload_labels, x.shape[0], context, require_labels)
            return x, y, names

        raise ValueError(
            f"{context} must provide COMPARE features or metric rows. Supported keys: "
            "member_features/nonmember_features, features, member_metrics/nonmember_metrics, metrics."
        )

    def _features_from_metric_rows(
        self,
        rows: Any,
        payload: Mapping[str, Any],
    ) -> Tuple[np.ndarray, List[str]]:
        normalized_rows = list(rows)
        alpha_values = list(payload.get("alpha_values", self.alpha_values))
        names = ["target_ndcg_10"] + [f"ndcg_diff_alpha_{alpha}" for alpha in alpha_values]
        features = np.zeros((len(normalized_rows), len(names)), dtype=np.float64)

        for row_index, row in enumerate(normalized_rows):
            if not isinstance(row, Mapping):
                raise TypeError("COMPARE metric rows must be mappings.")
            target_ndcg = self._read_metric(row, ("target_ndcg_10", "target_ndcg@10", "ndcg@10"))
            features[row_index, 0] = target_ndcg
            for alpha_index, alpha in enumerate(alpha_values, start=1):
                surrogate_ndcg = self._read_surrogate_metric(row, alpha)
                features[row_index, alpha_index] = target_ndcg - surrogate_ndcg
        return features, names

    def _read_metric(self, row: Mapping[str, Any], keys: Iterable[str]) -> float:
        for key in keys:
            if key in row:
                return float(row[key])
        raise KeyError(f"Missing COMPARE metric. Expected one of: {list(keys)}")

    def _read_surrogate_metric(self, row: Mapping[str, Any], alpha: float) -> float:
        alpha_text = str(alpha)
        candidate_keys = (
            f"surrogate_ndcg_10_alpha_{alpha_text}",
            f"surrogate_ndcg@10_alpha_{alpha_text}",
            f"surrogate_alpha_{alpha_text}_ndcg_10",
            f"surrogate_alpha_{alpha_text}_ndcg@10",
        )
        for key in candidate_keys:
            if key in row:
                return float(row[key])

        surrogate_by_alpha = row.get("surrogate_ndcg_10") or row.get("surrogate_ndcg@10")
        if isinstance(surrogate_by_alpha, Mapping):
            if alpha in surrogate_by_alpha:
                return float(surrogate_by_alpha[alpha])
            if alpha_text in surrogate_by_alpha:
                return float(surrogate_by_alpha[alpha_text])

        raise KeyError(f"Missing surrogate NDCG@10 metric for alpha={alpha}.")

    def _feature_names(self, payload: Mapping[str, Any], feature_dim: int) -> List[str]:
        names = payload.get("feature_names")
        if names is None:
            return self._default_feature_names(feature_dim)
        names = [str(name) for name in names]
        if len(names) != feature_dim:
            raise ValueError(
                f"feature_names length mismatch: expected {feature_dim}, got {len(names)}."
            )
        return names

    def _default_feature_names(self, feature_dim: int) -> List[str]:
        compare_names = ["target_ndcg_10"] + [
            f"ndcg_diff_alpha_{alpha}" for alpha in self.alpha_values
        ]
        if feature_dim == len(compare_names):
            return compare_names
        return [f"feature_{idx}" for idx in range(feature_dim)]

    def _coerce_labels(
        self,
        labels: Optional[Any],
        expected_size: int,
        context: str,
        require_labels: bool,
    ) -> Optional[np.ndarray]:
        if labels is None:
            if require_labels:
                raise ValueError(f"{context} requires membership_labels.")
            return None
        y = self._to_numpy_1d(labels).astype(np.int64)
        if y.shape[0] != expected_size:
            raise ValueError(
                f"{context} membership label count mismatch: expected {expected_size}, got {y.shape[0]}."
            )
        return y

    def _to_numpy_1d(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy().reshape(-1)
        return np.asarray(value).reshape(-1)

    def _to_numpy_2d(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}.")
        return array.astype(np.float64)


__all__ = ["CompareMIAAttack"]
