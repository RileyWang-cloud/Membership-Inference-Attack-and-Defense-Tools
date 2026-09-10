"""
Supplementary recommender-system defense demo.

This script evaluates recommender-list defenses against the existing Biased-MIA
recommender attack adapter.

Run examples
------------
python Defense/rec_privacy_defense_demo.py
python Defense/rec_privacy_defense_demo.py --defense shuffle
python Defense/rec_privacy_defense_demo.py --replacement-probability 0.5
python Defense/rec_privacy_defense_demo.py --alpha-pr 0.1
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.biased_mia import BiasedMIAAttack
from Attack.shadow_based import AttackInput
from Defense.base import DefenseInput
from Defense.rec_privacy_defenses import (
    PopularityRandomizationDefense,
    RecommendationListShuffleDefense,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_item_embeddings(num_items: int, dim: int) -> Dict[int, np.ndarray]:
    return {
        item_id: np.random.normal(0.0, 0.6, size=(dim,)).astype(np.float32)
        for item_id in range(num_items)
    }


def build_raw_rec_payload(
    start_user_id: int,
    num_users: int,
    item_embeddings: Dict[int, np.ndarray],
    member: bool,
    interaction_len: int = 6,
    recommendation_len: int = 6,
) -> Tuple[Dict[int, list], Dict[int, list]]:
    item_ids = np.asarray(list(item_embeddings.keys()))
    popular_items = item_ids[: len(item_ids) // 2]
    tail_items = item_ids[len(item_ids) // 2 :]

    interactions: Dict[int, list] = {}
    recommendations: Dict[int, list] = {}
    for user_id in range(start_user_id, start_user_id + num_users):
        interactions[user_id] = np.random.choice(
            popular_items,
            size=interaction_len,
            replace=True,
        ).astype(int).tolist()

        recommendation_pool = popular_items if member else tail_items
        recommendations[user_id] = np.random.choice(
            recommendation_pool,
            size=recommendation_len,
            replace=True,
        ).astype(int).tolist()

    return interactions, recommendations


def build_attack_input(
    samples: Dict[str, object],
    shadow_data: Dict[str, object],
    num_member: int,
    num_nonmember: int,
    item_embeddings: Dict[int, np.ndarray],
    epochs: int,
) -> AttackInput:
    return AttackInput(
        target_model=None,
        samples={**samples, "item_embeddings": item_embeddings},
        membership_labels=np.concatenate(
            [
                np.ones(num_member, dtype=np.int64),
                np.zeros(num_nonmember, dtype=np.int64),
            ],
            axis=0,
        ),
        shadow_data={**shadow_data, "item_embeddings": item_embeddings},
        config={
            "hidden_dims": [32, 8],
            "batch_size": 32,
            "epochs": epochs,
            "lr": 1e-2,
            "momentum": 0.7,
        },
    )


def run_biased_mia(attack_input: AttackInput, seed: int) -> Tuple[float, float]:
    set_seed(seed)
    attack = BiasedMIAAttack()
    output = attack.run(attack_input)
    evaluation = output.evaluation
    assert evaluation is not None
    return float(evaluation.accuracy or 0.0), float(evaluation.auroc or 0.0)


def print_result(name: str, accuracy: float, auroc: float) -> None:
    print(f"{name:<26} accuracy={accuracy:.4f} auroc={auroc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recommender MIA defense demo")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-items", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=100)
    parser.add_argument("--shadow-users", type=int, default=96)
    parser.add_argument("--target-users", type=int, default=40)
    parser.add_argument("--attack-epochs", type=int, default=20)
    parser.add_argument(
        "--defense",
        choices=("popularity", "shuffle"),
        default="popularity",
    )
    parser.add_argument("--replacement-probability", type=float, default=1.0)
    parser.add_argument("--alpha-pr", type=float, default=0.1)
    parser.add_argument("--candidate-pool-size", type=int, default=None)
    parser.add_argument("--shuffle-probability", type=float, default=1.0)
    args = parser.parse_args()

    set_seed(args.seed)
    item_embeddings = build_item_embeddings(args.num_items, args.embedding_dim)
    popular_items = list(range(args.num_items))

    shadow_member_interactions, shadow_member_recommendations = build_raw_rec_payload(
        0,
        args.shadow_users,
        item_embeddings,
        member=True,
    )
    shadow_nonmember_interactions, shadow_nonmember_recommendations = build_raw_rec_payload(
        1000,
        args.shadow_users,
        item_embeddings,
        member=False,
    )
    target_member_interactions, target_member_recommendations = build_raw_rec_payload(
        2000,
        args.target_users,
        item_embeddings,
        member=True,
    )
    target_nonmember_interactions, target_nonmember_recommendations = build_raw_rec_payload(
        3000,
        args.target_users,
        item_embeddings,
        member=False,
    )

    shadow_data = {
        "member_interactions": shadow_member_interactions,
        "nonmember_interactions": shadow_nonmember_interactions,
        "member_recommendations": shadow_member_recommendations,
        "nonmember_recommendations": shadow_nonmember_recommendations,
    }
    target_samples = {
        "member_interactions": target_member_interactions,
        "nonmember_interactions": target_nonmember_interactions,
        "member_recommendations": target_member_recommendations,
        "nonmember_recommendations": target_nonmember_recommendations,
    }

    baseline_input = build_attack_input(
        target_samples,
        shadow_data,
        args.target_users,
        args.target_users,
        item_embeddings,
        args.attack_epochs,
    )
    baseline_accuracy, baseline_auroc = run_biased_mia(baseline_input, args.seed)

    if args.defense == "shuffle":
        defense = RecommendationListShuffleDefense()
        defense_config = {
            "seed": args.seed,
            "shuffle_probability": args.shuffle_probability,
            "target_splits": ["member", "nonmember"],
        }
    else:
        defense = PopularityRandomizationDefense()
        defense_config = {
            "seed": args.seed,
            "alpha_pr": args.alpha_pr,
            "replacement_probability": args.replacement_probability,
            "candidate_pool_size": args.candidate_pool_size,
            "target_splits": ["nonmember"],
        }

    defense_output = defense.run(
        DefenseInput(
            samples=target_samples,
            auxiliary_data={"popular_items": popular_items},
            defense_config=defense_config,
            eval_config={},
        )
    )
    defended_samples = defense_output.transformed_data
    defended_input = build_attack_input(
        defended_samples,
        shadow_data,
        args.target_users,
        args.target_users,
        item_embeddings,
        args.attack_epochs,
    )
    defended_accuracy, defended_auroc = run_biased_mia(defended_input, args.seed)

    print("Biased-MIA against recommendation lists")
    print_result("baseline", baseline_accuracy, baseline_auroc)
    print_result(defense.name, defended_accuracy, defended_auroc)
    print(
        "changed_users="
        f"{defense_output.metadata['changed_users']}/{defense_output.metadata['total_users']}"
    )
    if args.defense == "shuffle":
        print(
            "shuffle_probability="
            f"{defense_output.metadata['shuffle_probability']} "
            "mean_rank_displacement="
            f"{defense_output.metadata['mean_rank_displacement']:.2f}"
        )
    else:
        print(
            "alpha_pr="
            f"{defense_output.metadata['alpha_pr']} "
            "mean_candidate_size="
            f"{defense_output.metadata['mean_candidate_size']:.1f}"
        )


if __name__ == "__main__":
    main()
