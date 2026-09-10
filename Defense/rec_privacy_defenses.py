"""
Recommendation-system privacy defenses for MIA experiments.

The defenses implemented here are model-agnostic output/data-processing
wrappers for recommender-system MIA settings. They can be used with both
standard and sequential recommenders as long as the experiment exposes
user-level recommendation lists.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

import numpy as np

from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


RecommendationPayload = Dict[str, Any]


class PopularityRandomizationDefense(BaseDefense):
    """
    Paper-style Popularity Randomization for recommendation lists.

    defense_mode: data_processing / inference_time
    Required:
        - samples or signals containing recommendation lists
    Optional:
        - auxiliary_data["item_popularity"], ["popular_items"], or interactions
    Main output:
        - transformed_data with randomized recommendation lists

    Supported payload keys include the partitioned recommender-MIA format:
        member_recommendations, nonmember_recommendations
    and the combined format:
        recommendations

    Paper parameters:
        Ssorted = items sorted by popularity
        alpha_pr = Nrec / Ncand
        Scand = first Ncand items in Ssorted
        Rout = random sample from Scand with |Rout| = Nrec
    """

    name = "popularity_randomization"
    defense_family = "recommender_privacy"
    defense_mode = "data_processing"
    supported_model_types = ["recommender", "sequential_recommender"]
    required_input_keys = ["samples or signals"]
    optional_input_keys = ["auxiliary_data.item_popularity", "auxiliary_data.popular_items"]

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        payload = self._resolve_payload(defense_input)
        transformed = deepcopy(payload)
        config = self._merge_config(defense_input.defense_config)
        rng = np.random.default_rng(config["seed"])

        sorted_items = self._resolve_sorted_items(defense_input, payload)
        changed_users = 0
        total_users = 0
        candidate_sizes: List[int] = []

        partition_keys = self._partition_recommendation_keys(transformed)
        if partition_keys:
            target_splits = set(config["target_splits"])
            for split, key in partition_keys.items():
                if split not in target_splits:
                    continue
                randomized, split_total, split_changed, split_candidate_sizes = self._randomize_recommendation_mapping(
                    transformed[key],
                    sorted_items,
                    rng,
                    config,
                    interactions=self._matching_interactions(transformed, split),
                )
                transformed[key] = randomized
                total_users += split_total
                changed_users += split_changed
                candidate_sizes.extend(split_candidate_sizes)
        elif "recommendations" in transformed:
            randomized, total_users, changed_users, candidate_sizes = self._randomize_recommendation_mapping(
                transformed["recommendations"],
                sorted_items,
                rng,
                config,
                interactions=transformed.get("interactions"),
            )
            transformed["recommendations"] = randomized
        else:
            raise ValueError(
                "Recommendation payload must contain member/nonmember recommendation "
                "lists or a combined 'recommendations' field."
            )

        metadata = {
            "defense_name": self.name,
            "defense_family": self.defense_family,
            "defense_mode": self.defense_mode,
            "changed_users": changed_users,
            "total_users": total_users,
            "replacement_probability": config["replacement_probability"],
            "alpha_pr": config["alpha_pr"],
            "candidate_pool_size": config["candidate_pool_size"],
            "sorted_item_count": len(sorted_items),
            "mean_candidate_size": float(np.mean(candidate_sizes)) if candidate_sizes else 0.0,
        }
        return DefenseOutput(
            transformed_data=transformed,
            artifacts={"Ssorted": list(sorted_items)},
            intermediate_outputs={"candidate_sizes": candidate_sizes},
            metadata=metadata,
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        utility_metrics: Dict[str, float] = {}
        eval_config = defense_input.eval_config or {}
        utility_fn = eval_config.get("utility_fn")
        if callable(utility_fn):
            result = utility_fn(defense_output.transformed_data, defense_input)
            if isinstance(result, Mapping):
                utility_metrics.update({str(k): float(v) for k, v in result.items()})

        privacy_metrics = {
            "changed_user_ratio": float(
                defense_output.metadata.get("changed_users", 0)
                / max(1, defense_output.metadata.get("total_users", 0))
            )
        }
        return DefenseEvaluationResult(
            utility_metrics=utility_metrics or None,
            privacy_metrics=privacy_metrics,
            efficiency_metrics=None,
            extra_metrics=None,
        )

    def _merge_config(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        merged = {
            "seed": 2026,
            "alpha_pr": 0.1,
            "replacement_probability": 1.0,
            "candidate_pool_size": None,
            "target_splits": ["nonmember"],
            "sample_without_replacement": True,
            "exclude_seen_items": False,
        }
        merged.update(dict(config))
        if isinstance(merged["target_splits"], str):
            merged["target_splits"] = [merged["target_splits"]]
        alpha_pr = float(merged["alpha_pr"])
        if alpha_pr <= 0.0 or alpha_pr > 1.0:
            raise ValueError("alpha_pr must be in the interval (0, 1].")
        return merged

    def _resolve_payload(self, defense_input: DefenseInput) -> RecommendationPayload:
        if isinstance(defense_input.samples, dict):
            return defense_input.samples
        if isinstance(defense_input.signals, dict):
            return defense_input.signals
        raise ValueError("PopularityRandomizationDefense requires dict samples or signals.")

    def _partition_recommendation_keys(self, payload: Mapping[str, Any]) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        for split in ("member", "nonmember"):
            key = f"{split}_recommendations"
            if key in payload:
                keys[split] = key
        return keys

    def _matching_interactions(self, payload: Mapping[str, Any], split: str) -> Optional[Any]:
        return payload.get(f"{split}_interactions")

    def _resolve_sorted_items(
        self,
        defense_input: DefenseInput,
        payload: Mapping[str, Any],
    ) -> List[Any]:
        auxiliary = defense_input.auxiliary_data or {}
        if "popular_items" in auxiliary:
            items = list(auxiliary["popular_items"])
        elif "item_popularity" in auxiliary:
            popularity = auxiliary["item_popularity"]
            if isinstance(popularity, Mapping):
                items = [
                    item
                    for item, _ in sorted(
                        popularity.items(),
                        key=lambda pair: pair[1],
                        reverse=True,
                    )
                ]
            else:
                items = list(popularity)
        else:
            items = self._derive_popular_items_from_payload(payload)

        if not items:
            raise ValueError("Unable to derive a non-empty popularity item pool.")

        return list(items)

    def _derive_popular_items_from_payload(self, payload: Mapping[str, Any]) -> List[Any]:
        counter: Counter[Any] = Counter()
        interaction_keys = (
            "member_interactions",
            "nonmember_interactions",
            "interactions",
        )
        recommendation_keys = (
            "member_recommendations",
            "nonmember_recommendations",
            "recommendations",
        )
        for key in interaction_keys:
            if key in payload:
                counter.update(self._iter_items(payload[key]))
        if not counter:
            for key in recommendation_keys:
                if key in payload:
                    counter.update(self._iter_items(payload[key]))
        return [item for item, _ in counter.most_common()]

    def _iter_items(self, sequences: Any) -> Iterable[Any]:
        if isinstance(sequences, Mapping):
            iterable = sequences.values()
        else:
            iterable = sequences
        for seq in iterable:
            for item in seq:
                yield item

    def _randomize_recommendation_mapping(
        self,
        recommendations: Any,
        sorted_items: Sequence[Any],
        rng: np.random.Generator,
        config: Mapping[str, Any],
        interactions: Optional[Any] = None,
    ) -> tuple[Any, int, int, List[int]]:
        entries = self._normalize_user_sequences(recommendations)
        seen_by_user = self._normalize_user_sequence_dict(interactions) if interactions is not None else {}
        randomized_entries = []
        candidate_sizes = []
        changed = 0

        for user_id, rec_items in entries:
            if rng.random() > float(config["replacement_probability"]):
                randomized_entries.append((user_id, list(rec_items)))
                continue

            candidates = self._select_candidates(sorted_items, len(rec_items), config)
            if config["exclude_seen_items"] and user_id in seen_by_user:
                seen = set(seen_by_user[user_id])
                candidates = [item for item in candidates if item not in seen]
            if not candidates:
                candidates = self._select_candidates(sorted_items, len(rec_items), config)
            candidate_sizes.append(len(candidates))

            new_items = self._sample_items(
                candidates,
                len(rec_items),
                rng,
                without_replacement=bool(config["sample_without_replacement"]),
            )
            randomized_entries.append((user_id, new_items))
            changed += 1

        return (
            self._restore_user_sequences(recommendations, randomized_entries),
            len(entries),
            changed,
            candidate_sizes,
        )

    def _select_candidates(
        self,
        sorted_items: Sequence[Any],
        recommendation_count: int,
        config: Mapping[str, Any],
    ) -> List[Any]:
        if recommendation_count <= 0:
            return []
        candidate_pool_size = config.get("candidate_pool_size")
        if candidate_pool_size is None:
            candidate_pool_size = int(np.ceil(recommendation_count / float(config["alpha_pr"])))
        candidate_pool_size = max(recommendation_count, int(candidate_pool_size))
        candidate_pool_size = min(candidate_pool_size, len(sorted_items))
        return list(sorted_items[:candidate_pool_size])

    def _sample_items(
        self,
        candidates: Sequence[Any],
        size: int,
        rng: np.random.Generator,
        without_replacement: bool,
    ) -> List[Any]:
        if size <= 0:
            return []
        replace = not without_replacement or len(candidates) < size
        indices = rng.choice(len(candidates), size=size, replace=replace)
        return [candidates[int(idx)] for idx in np.asarray(indices).reshape(-1)]

    def _normalize_user_sequences(self, value: Any) -> List[tuple[Any, List[Any]]]:
        if isinstance(value, Mapping):
            return [(user_id, list(items)) for user_id, items in value.items()]
        if isinstance(value, (list, tuple)):
            return [(idx, list(items)) for idx, items in enumerate(value)]
        raise TypeError("recommendations must be a mapping or a sequence of item-id lists.")

    def _normalize_user_sequence_dict(self, value: Any) -> Dict[Any, List[Any]]:
        if value is None:
            return {}
        return {user_id: items for user_id, items in self._normalize_user_sequences(value)}

    def _restore_user_sequences(self, original: Any, entries: List[tuple[Any, List[Any]]]) -> Any:
        if isinstance(original, MutableMapping):
            return {user_id: items for user_id, items in entries}
        if isinstance(original, tuple):
            return tuple(items for _, items in entries)
        return [items for _, items in entries]


class SequentialRecommendationRandomizationDefense(PopularityRandomizationDefense):
    """
    Alias with sequential-recommender metadata.

    Sequential MIA experiments can pass the exposed next-item/top-k lists through
    the same payload format used by standard recommender attacks.
    """

    name = "sequential_recommendation_randomization"
    supported_model_types = ["sequential_recommender"]


class RecommendationListShuffleDefense(BaseDefense):
    """
    Shuffle exposed recommendation lists to remove reliable rank information.

    defense_mode: inference_time / data_processing
    Required:
        - samples or signals containing recommendation lists
    Optional:
        - target_model, when a protected predictor wrapper is needed
    Main output:
        - transformed_data / protected_outputs with shuffled recommendation lists

    This matches the A.5 defense idea for recommender MIA: keep the recommended
    item set unchanged, but make item order untrustworthy so attacks depending
    on rank-position signals lose information.
    """

    name = "recommendation_list_shuffle"
    defense_family = "recommender_privacy"
    defense_mode = "inference_time"
    supported_model_types = ["recommender", "sequential_recommender"]
    required_input_keys = ["samples or signals or target_model"]
    optional_input_keys = ["defense_config.target_splits", "defense_config.shuffle_probability"]

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        config = self._merge_config(defense_input.defense_config)
        rng = np.random.default_rng(config["seed"])
        protected_predictor = None
        if defense_input.target_model is not None:
            protected_predictor = _ShuffledRecommendationPredictor(
                defense_input.target_model,
                config=config,
            )

        payload = self._resolve_payload(defense_input)
        if payload is None:
            if protected_predictor is None:
                raise ValueError(
                    "RecommendationListShuffleDefense requires dict samples/signals "
                    "or a target_model to wrap."
                )
            return DefenseOutput(
                protected_predictor=protected_predictor,
                metadata={
                    "defense_name": self.name,
                    "defense_family": self.defense_family,
                    "defense_mode": self.defense_mode,
                    "shuffle_probability": config["shuffle_probability"],
                    "changed_users": 0,
                    "total_users": 0,
                },
            )

        transformed = deepcopy(payload)
        changed_users = 0
        total_users = 0
        rank_displacements: List[float] = []

        partition_keys = self._partition_recommendation_keys(transformed)
        if partition_keys:
            target_splits = set(config["target_splits"])
            for split, key in partition_keys.items():
                if split not in target_splits:
                    continue
                shuffled, split_total, split_changed, split_displacements = self._shuffle_recommendation_mapping(
                    transformed[key],
                    rng,
                    config,
                )
                transformed[key] = shuffled
                total_users += split_total
                changed_users += split_changed
                rank_displacements.extend(split_displacements)
        elif "recommendations" in transformed:
            shuffled, total_users, changed_users, rank_displacements = self._shuffle_recommendation_mapping(
                transformed["recommendations"],
                rng,
                config,
            )
            transformed["recommendations"] = shuffled
        else:
            raise ValueError(
                "Recommendation payload must contain member/nonmember recommendation "
                "lists or a combined 'recommendations' field."
            )

        metadata = {
            "defense_name": self.name,
            "defense_family": self.defense_family,
            "defense_mode": self.defense_mode,
            "changed_users": changed_users,
            "total_users": total_users,
            "shuffle_probability": config["shuffle_probability"],
            "target_splits": list(config["target_splits"]),
            "mean_rank_displacement": float(np.mean(rank_displacements)) if rank_displacements else 0.0,
        }
        return DefenseOutput(
            protected_predictor=protected_predictor,
            protected_outputs=transformed,
            transformed_data=transformed,
            intermediate_outputs={"rank_displacements": rank_displacements},
            metadata=metadata,
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        privacy_metrics = {
            "changed_user_ratio": float(
                defense_output.metadata.get("changed_users", 0)
                / max(1, defense_output.metadata.get("total_users", 0))
            ),
            "mean_rank_displacement": float(
                defense_output.metadata.get("mean_rank_displacement", 0.0)
            ),
        }
        return DefenseEvaluationResult(
            utility_metrics={"item_set_preservation": 1.0},
            privacy_metrics=privacy_metrics,
            efficiency_metrics=None,
            extra_metrics=None,
        )

    def _merge_config(self, config: Mapping[str, Any]) -> Dict[str, Any]:
        merged = {
            "seed": 2026,
            "shuffle_probability": 1.0,
            "target_splits": ["member", "nonmember"],
        }
        merged.update(dict(config))
        if isinstance(merged["target_splits"], str):
            merged["target_splits"] = [merged["target_splits"]]
        probability = float(merged["shuffle_probability"])
        if probability < 0.0 or probability > 1.0:
            raise ValueError("shuffle_probability must be in the interval [0, 1].")
        merged["shuffle_probability"] = probability
        return merged

    def _resolve_payload(self, defense_input: DefenseInput) -> Optional[RecommendationPayload]:
        if isinstance(defense_input.samples, dict):
            return defense_input.samples
        if isinstance(defense_input.signals, dict):
            return defense_input.signals
        return None

    def _partition_recommendation_keys(self, payload: Mapping[str, Any]) -> Dict[str, str]:
        keys: Dict[str, str] = {}
        for split in ("member", "nonmember"):
            key = f"{split}_recommendations"
            if key in payload:
                keys[split] = key
        return keys

    def _shuffle_recommendation_mapping(
        self,
        recommendations: Any,
        rng: np.random.Generator,
        config: Mapping[str, Any],
    ) -> tuple[Any, int, int, List[float]]:
        entries = self._normalize_user_sequences(recommendations)
        shuffled_entries = []
        rank_displacements = []
        changed = 0

        for user_id, rec_items in entries:
            original = list(rec_items)
            if len(original) <= 1 or rng.random() > float(config["shuffle_probability"]):
                shuffled_entries.append((user_id, original))
                continue

            permutation = rng.permutation(len(original))
            shuffled = [original[int(idx)] for idx in permutation]
            shuffled_entries.append((user_id, shuffled))
            if shuffled != original:
                changed += 1
                rank_displacements.append(
                    float(np.mean(np.abs(permutation - np.arange(len(original)))))
                )

        return (
            self._restore_user_sequences(recommendations, shuffled_entries),
            len(entries),
            changed,
            rank_displacements,
        )

    def _normalize_user_sequences(self, value: Any) -> List[tuple[Any, List[Any]]]:
        if isinstance(value, Mapping):
            return [(user_id, list(items)) for user_id, items in value.items()]
        if isinstance(value, (list, tuple)):
            return [(idx, list(items)) for idx, items in enumerate(value)]
        raise TypeError("recommendations must be a mapping or a sequence of item-id lists.")

    def _restore_user_sequences(self, original: Any, entries: List[tuple[Any, List[Any]]]) -> Any:
        if isinstance(original, MutableMapping):
            return {user_id: items for user_id, items in entries}
        if isinstance(original, tuple):
            return tuple(items for _, items in entries)
        return [items for _, items in entries]


class _ShuffledRecommendationPredictor:
    """Thin predictor wrapper for common recommender APIs."""

    def __init__(self, target_model: Any, config: Mapping[str, Any]) -> None:
        self.target_model = target_model
        self.config = dict(config)
        self.rng = np.random.default_rng(self.config["seed"])

    def recommend(self, *args: Any, **kwargs: Any) -> Any:
        recommendations = self.target_model.recommend(*args, **kwargs)
        return self._shuffle_output(recommendations)

    def recommend_for_user(self, *args: Any, **kwargs: Any) -> Any:
        recommendations = self.target_model.recommend_for_user(*args, **kwargs)
        return self._shuffle_output(recommendations)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self.target_model(*args, **kwargs)
        return self._shuffle_output(output)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.target_model, name)

    def _shuffle_output(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: self._shuffle_sequence(items)
                for key, items in value.items()
            }
        if isinstance(value, tuple) and value and isinstance(value[0], (list, tuple)):
            return tuple(self._shuffle_sequence(items) for items in value)
        if isinstance(value, tuple):
            return tuple(self._shuffle_sequence(value))
        if isinstance(value, list) and value and isinstance(value[0], (list, tuple)):
            return [self._shuffle_sequence(items) for items in value]
        if isinstance(value, list):
            return self._shuffle_sequence(value)
        return value

    def _shuffle_sequence(self, items: Any) -> Any:
        original = list(items)
        if len(original) <= 1 or self.rng.random() > float(self.config["shuffle_probability"]):
            return original
        permutation = self.rng.permutation(len(original))
        return [original[int(idx)] for idx in permutation]


__all__ = [
    "PopularityRandomizationDefense",
    "RecommendationListShuffleDefense",
    "SequentialRecommendationRandomizationDefense",
]
