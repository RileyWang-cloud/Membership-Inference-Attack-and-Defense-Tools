"""
Shadow-Free Membership Inference Attack (IJCAI-24), wrapped with the project's
minimal attack interface.

Reimplemented from scratch following ``ATTACK_INTERFACE_DESIGN_ZH.md``; the
earlier ``shadow_free_mia_core.py`` copy was removed as unreliable. The attack
needs neither shadow models nor target-model queries — only pretrained item
embeddings plus a popularity ranking.

Algorithm
---------
Global baseline vector (built once in ``fit``)::

    baseline = mean embedding of the top-k most popular items

Per user::

    interaction_vector    = mean embedding of the user's interaction history
    recommendation_vector = mean embedding of the user's recommendation list
    S1 = sim(interaction_vector, recommendation_vector)
    S2 = sim(recommendation_vector, baseline)
    sim(v1, v2) = 1 / (||v1 - v2|| + eps)

A member's recommendations track their own taste (S1 > S2); a non-member's
recommendations resemble the popularity baseline (S1 < S2).

Unified convention (design doc §7.1, §10.3)::

    membership_score = S1 - S2   (higher -> more likely member)
    membership_pred  = 1 if S1 > S2 else 0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from Attack.base import AttackInput, AttackOutput, BaseAttack


@dataclass(frozen=True)
class UserRecord:
    """A single user to attack in the recommender setting.

    ``samples`` passed to ``AttackInput`` is an iterable of these (or of
    mappings with the same keys). Records are scored in iteration order.
    """

    user_id: Any
    interaction_items: Sequence[int]
    recommended_items: Sequence[int]


class ShadowFreeMIAAttack(BaseAttack):
    """
    Shadow-Free Membership Inference Attack against recommender systems.

    No shadow training, no target-model query. Needs only pretrained item
    embeddings plus a popularity ranking.

    Required AttackInput fields
    ---------------------------
    - samples
        Iterable of :class:`UserRecord` (or mappings with keys ``user_id``,
        ``interaction_items``, ``recommended_items``). One record per user.
    - signals["item_embeddings"]
        ``Mapping[item_id -> embedding]``. Embeddings may be numpy arrays, torch
        tensors, or anything ``numpy.asarray`` accepts. Latent dim is read from
        the embeddings themselves.
    - reference_data["popular_items"]
        Item ids sorted by popularity (most popular first); used to build the
        baseline vector in ``fit``.

    Optional
    --------
    - reference_data["baseline_vector"]
        Precomputed baseline; if present, ``fit`` skips aggregation.
    - config: ``{"top_k": 100, "eps": 1e-8}`` (override the constructor).

    Main output
    -----------
    - ``membership_scores`` (= S1 - S2), ``membership_preds``, plus S1 / S2 and
      item coverage in ``intermediate_outputs``.
    """

    def __init__(self, top_k: int = 100, eps: float = 1e-8) -> None:
        self.top_k = int(top_k)
        self.eps = float(eps)
        self.baseline_vector: Optional[np.ndarray] = None

    # ------------------------------------------------------------------- fit
    def fit(self, attack_input: AttackInput) -> "ShadowFreeMIAAttack":
        reference_data = attack_input.reference_data or {}
        top_k = int(attack_input.config.get("top_k", self.top_k))

        if "baseline_vector" in reference_data:
            self.baseline_vector = _to_numpy_vec(reference_data["baseline_vector"])
            return self

        item_embeddings = _require_item_embeddings(attack_input)
        popular_items = reference_data.get("popular_items")
        if not popular_items:
            raise ValueError(
                "ShadowFreeMIAAttack.fit requires reference_data['popular_items'] "
                "(item ids sorted by popularity) or a precomputed "
                "reference_data['baseline_vector']."
            )

        id_to_row, table = _build_embedding_table(item_embeddings)
        baseline = _mean_embedding(popular_items, id_to_row, table, top_k=top_k)
        if baseline is None:
            raise ValueError(
                "None of the top_k popular_items were found in "
                "signals['item_embeddings']; cannot build baseline vector."
            )
        self.baseline_vector = baseline
        return self

    # ----------------------------------------------------------------- infer
    def infer(self, attack_input: AttackInput) -> AttackOutput:
        if self.baseline_vector is None:
            raise RuntimeError("ShadowFreeMIAAttack must be fitted before infer().")

        eps = float(attack_input.config.get("eps", self.eps))
        item_embeddings = _require_item_embeddings(attack_input)
        id_to_row, table = _build_embedding_table(item_embeddings)
        records = _coerce_samples(attack_input.samples)

        user_ids: List[Any] = []
        s1_scores: List[float] = []
        s2_scores: List[float] = []
        membership_scores: List[float] = []
        membership_preds: List[int] = []
        known_hits = 0
        known_total = 0

        for record in records:
            interaction_vec = _mean_embedding(record.interaction_items, id_to_row, table)
            recommendation_vec = _mean_embedding(record.recommended_items, id_to_row, table)

            known_hits += _count_known(record.interaction_items, id_to_row)
            known_hits += _count_known(record.recommended_items, id_to_row)
            known_total += len(record.interaction_items) + len(record.recommended_items)

            if interaction_vec is None or recommendation_vec is None:
                raise ValueError(
                    f"User {record.user_id!r} has none of its interaction/"
                    "recommendation items present in signals['item_embeddings']; "
                    "cannot score."
                )

            s1 = _inverse_euclidean(interaction_vec, recommendation_vec, eps)
            s2 = _inverse_euclidean(recommendation_vec, self.baseline_vector, eps)
            score = s1 - s2
            pred = 1 if s1 > s2 else 0

            user_ids.append(record.user_id)
            s1_scores.append(s1)
            s2_scores.append(s2)
            membership_scores.append(score)
            membership_preds.append(pred)

        coverage = (known_hits / known_total) if known_total else 0.0
        return AttackOutput(
            membership_scores=np.asarray(membership_scores, dtype=np.float64),
            membership_preds=np.asarray(membership_preds, dtype=np.int64),
            intermediate_outputs={
                "user_ids": user_ids,
                "S1": np.asarray(s1_scores, dtype=np.float64),
                "S2": np.asarray(s2_scores, dtype=np.float64),
                "item_coverage": coverage,
                "baseline_vector": self.baseline_vector,
            },
            metadata={
                "attack_name": "shadow_free_mia",
                "top_k": self.top_k,
                "latent_dim": int(table.shape[1]),
                "num_users": len(records),
            },
        )


# --------------------------------------------------------------------- helpers
def _require_item_embeddings(attack_input: AttackInput) -> Mapping[int, Any]:
    signals = attack_input.signals or {}
    item_embeddings = signals.get("item_embeddings")
    if item_embeddings is None:
        raise ValueError(
            "ShadowFreeMIAAttack requires signals['item_embeddings'] "
            "(Mapping[item_id -> embedding])."
        )
    return item_embeddings


def _coerce_samples(samples: Any) -> List[UserRecord]:
    if samples is None:
        raise ValueError("ShadowFreeMIAAttack requires `samples` (user records).")
    records: List[UserRecord] = []
    for raw in samples:
        if isinstance(raw, UserRecord):
            records.append(raw)
        elif isinstance(raw, Mapping):
            records.append(
                UserRecord(
                    user_id=raw.get("user_id"),
                    interaction_items=list(raw.get("interaction_items", [])),
                    recommended_items=list(raw.get("recommended_items", [])),
                )
            )
        else:
            raise TypeError(
                "Each entry in `samples` must be a UserRecord or a mapping with "
                "keys user_id / interaction_items / recommended_items."
            )
    if not records:
        raise ValueError("`samples` must contain at least one user record.")
    return records


def _build_embedding_table(
    item_embeddings: Mapping[int, Any],
) -> Tuple[Dict[int, int], np.ndarray]:
    """Stack embeddings into a matrix.

    Returns ``(id_to_row, table)`` where ``table[id_to_row[item_id]]`` is the
    embedding for ``item_id``. The latent dimension is read from the embeddings.
    """
    ids = list(item_embeddings.keys())
    if not ids:
        raise ValueError("signals['item_embeddings'] is empty.")
    vectors = [_to_numpy_vec(item_embeddings[i]) for i in ids]
    table = np.stack(vectors, axis=0)
    if table.ndim != 2:
        raise ValueError(f"Expected 1-D item embeddings; got shape {table.shape}.")
    id_to_row = {int(item_id): row for row, item_id in enumerate(ids)}
    return id_to_row, table


def _mean_embedding(
    item_ids: Sequence[int],
    id_to_row: Mapping[int, int],
    table: np.ndarray,
    top_k: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Mean embedding of the given item ids (restricted to known ids).

    When ``top_k`` is set, only the first ``top_k`` ids are considered (used for
    the popularity baseline). Returns ``None`` if no id is known.
    """
    requested = list(item_ids)
    if top_k is not None:
        requested = requested[:top_k]
    if not requested:
        return None
    rows = [id_to_row[i] for i in requested if i in id_to_row]
    if not rows:
        return None
    return table[rows].astype(np.float64).mean(axis=0)


def _count_known(item_ids: Sequence[int], id_to_row: Mapping[int, int]) -> int:
    return sum(1 for i in item_ids if i in id_to_row)


def _to_numpy_vec(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):  # torch tensor
        value = value.detach().cpu().numpy()
    return np.asarray(value).astype(np.float64).reshape(-1)


def _inverse_euclidean(vec1: np.ndarray, vec2: np.ndarray, eps: float) -> float:
    diff = vec1 - vec2
    distance = float(np.sqrt(np.dot(diff, diff)))
    return 1.0 / (distance + eps)


__all__ = ["UserRecord", "ShadowFreeMIAAttack"]
