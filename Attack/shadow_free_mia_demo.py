"""
End-to-end demo for the Shadow-Free MIA using the project's minimal attack
interface.

We build a toy recommender scenario where the attack should clearly separate
members from non-members, then run it through ``ShadowFreeMIAAttack.run`` and
print the unified evaluation.

Scenario
--------
- A "popular" cluster of items around the origin (these define the baseline).
- Member users have a personal taste cluster far from the origin; both their
  interaction history and their recommendations are drawn from that cluster, so
  the recommendations track their taste (S1 high, S2 low).
- Non-member users have a personal taste cluster too, but their recommendations
  are drawn from the popular cluster, so the recommendations look like the
  baseline (S1 low, S2 high).

This makes the membership signal cleanly positive for members and negative for
non-members.

Run
---
python Attack/shadow_free_mia_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.base import AttackInput
from Attack.shadow_free_mia import ShadowFreeMIAAttack, UserRecord


NUM_LATENT = 16
NUM_POPULAR = 30
NUM_MEMBERS = 50
NUM_NONMEMBERS = 50
ITEMS_PER_LIST = 5
CLUSTER_STD = 0.5
SEED = 42


def build_scenario(rng: np.random.Generator):
    """Build item embeddings, popular ranking, user records and labels.

    Returns ``(item_embeddings, popular_items, records, labels)`` where labels
    align with records (1 = member, 0 = non-member).
    """
    item_embeddings: dict[int, np.ndarray] = {}
    popular_items: List[int] = []

    # Popular cluster around the origin.
    pop_center = np.zeros(NUM_LATENT)
    for item_id in range(NUM_POPULAR):
        item_embeddings[item_id] = rng.normal(pop_center, CLUSTER_STD)
        popular_items.append(item_id)

    records: List[UserRecord] = []
    labels: List[int] = []
    next_item_id = NUM_POPULAR

    def allocate_cluster(center: np.ndarray, count: int) -> List[int]:
        nonlocal next_item_id
        ids = []
        for _ in range(count):
            item_embeddings[next_item_id] = rng.normal(center, CLUSTER_STD)
            ids.append(next_item_id)
            next_item_id += 1
        return ids

    # Members: interaction & recommendations both from the personal cluster.
    for _ in range(NUM_MEMBERS):
        center = _random_direction(rng, NUM_LATENT) * 10.0
        niche_pool = allocate_cluster(center, ITEMS_PER_LIST * 2)
        interaction = niche_pool[:ITEMS_PER_LIST]
        recommendation = niche_pool[ITEMS_PER_LIST:]
        records.append(
            UserRecord(
                user_id=f"member_{len(records)}",
                interaction_items=interaction,
                recommended_items=recommendation,
            )
        )
        labels.append(1)

    # Non-members: interaction from personal cluster, recommendations = popular.
    for _ in range(NUM_NONMEMBERS):
        center = _random_direction(rng, NUM_LATENT) * 10.0
        interaction = allocate_cluster(center, ITEMS_PER_LIST)
        recommendation = rng.choice(popular_items, size=ITEMS_PER_LIST, replace=False).tolist()
        records.append(
            UserRecord(
                user_id=f"nonmember_{len(records)}",
                interaction_items=interaction,
                recommended_items=recommendation,
            )
        )
        labels.append(0)

    return item_embeddings, popular_items, records, np.asarray(labels, dtype=np.int64)


def _random_direction(rng: np.random.Generator, dim: int) -> np.ndarray:
    vec = rng.normal(np.zeros(dim), 1.0)
    return vec / np.linalg.norm(vec)


def main() -> None:
    rng = np.random.default_rng(SEED)
    item_embeddings, popular_items, records, labels = build_scenario(rng)

    attack_input = AttackInput(
        target_model=None,
        samples=records,
        signals={"item_embeddings": item_embeddings},
        reference_data={"popular_items": popular_items},
        membership_labels=labels,
        config={"top_k": NUM_POPULAR},
    )

    output = ShadowFreeMIAAttack().run(attack_input)

    scores = np.asarray(output.membership_scores, dtype=np.float64)
    preds = np.asarray(output.membership_preds, dtype=np.int64)
    members = labels == 1

    print("=" * 64)
    print("Shadow-Free MIA demo")
    print("=" * 64)
    print(f"users: {len(records)} (members={int(members.sum())}, "
          f"non-members={int((~members).sum())})")
    print(f"item embeddings: {len(item_embeddings)} (latent_dim={NUM_LATENT})")
    print(f"item coverage: {output.intermediate_outputs['item_coverage']:.3f}")
    print(f"mean score  members    : {scores[members].mean():+.4f}")
    print(f"mean score  non-members: {scores[~members].mean():+.4f}")
    print(f"pred accuracy (vs labels): {(preds == labels).mean():.3f}")
    print("-" * 64)
    print(f"evaluation: {output.evaluation}")
    print("=" * 64)

    # Self-checks (design doc §7.1, §10.3: higher score -> more likely member).
    assert scores[members].mean() > scores[~members].mean(), (
        "member mean score should exceed non-member mean score"
    )
    assert output.evaluation is not None and output.evaluation.auroc is not None
    assert output.evaluation.auroc > 0.9, (
        f"AUROC expected > 0.9 on the toy scenario, got {output.evaluation.auroc:.3f}"
    )
    # Preds must be consistent with the score sign (S1 - S2).
    assert np.array_equal(preds, (scores > 0).astype(np.int64)), (
        "membership_preds must equal (membership_scores > 0)"
    )
    print("All self-checks passed.")


if __name__ == "__main__":
    main()
