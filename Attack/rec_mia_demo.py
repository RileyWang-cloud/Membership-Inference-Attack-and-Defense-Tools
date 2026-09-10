"""
Minimal unified demo for recommender-system MIA attacks.

This script builds synthetic attack inputs and runs:
1. ME-MIA
2. Biased-MIA
3. DL-MIA
4. COMPARE

The goal is to demonstrate how all four methods are called through the
project's unified `AttackInput -> AttackOutput` interface.

Run examples
------------
python Attack/rec_mia_demo.py
python Attack/rec_mia_demo.py --method me
python Attack/rec_mia_demo.py --method biased
python Attack/rec_mia_demo.py --method dl
python Attack/rec_mia_demo.py --method compare
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
from Attack.compare_mia import CompareMIAAttack
from Attack.dl_mia import DLMIAAttack
from Attack.me_mia import MEMIAAttack
from Attack.shadow_based import AttackInput


def set_seed(seed: int = 2026) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_me_mia_scores(
    start_user_id: int,
    num_users: int,
    feature_dim: int,
    member: bool,
    seq_len_range: Tuple[int, int] = (4, 7),
) -> Dict[int, Dict[str, np.ndarray]]:
    score_dict: Dict[int, Dict[str, np.ndarray]] = {}
    signal_dims = slice(64, feature_dim)
    signal_mean = 1.2 if member else -1.2

    for user_id in range(start_user_id, start_user_id + num_users):
        seq_len = np.random.randint(seq_len_range[0], seq_len_range[1] + 1)
        features = np.random.normal(0.0, 0.35, size=(seq_len, feature_dim)).astype(np.float32)
        features[:, signal_dims] += signal_mean
        end = np.arange(seq_len, dtype=np.int64)
        score_dict[user_id] = {
            "features": features,
            "end": end,
        }
    return score_dict


def build_biased_vectors(num_users: int, dim: int, member: bool) -> np.ndarray:
    mean = 1.0 if member else -1.0
    vectors = np.random.normal(loc=mean, scale=0.45, size=(num_users, dim)).astype(np.float32)
    return vectors


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
    favored = item_ids[: len(item_ids) // 2]
    nonfavored = item_ids[len(item_ids) // 2 :]

    interactions: Dict[int, list] = {}
    recommendations: Dict[int, list] = {}
    for user_id in range(start_user_id, start_user_id + num_users):
        if member:
            interaction_pool = favored
            recommendation_pool = favored
        else:
            interaction_pool = favored
            recommendation_pool = nonfavored

        interactions[user_id] = np.random.choice(
            interaction_pool,
            size=interaction_len,
            replace=True,
        ).astype(int).tolist()
        recommendations[user_id] = np.random.choice(
            recommendation_pool,
            size=recommendation_len,
            replace=True,
        ).astype(int).tolist()
    return interactions, recommendations


def build_dl_features(num_users: int, member: bool) -> Dict[str, np.ndarray]:
    vector_mean = 0.9 if member else -0.9
    semantic_mean = 1.1 if member else -1.1
    syntax_mean = 0.6 if member else -0.6

    return {
        "vector": np.random.normal(vector_mean, 0.4, size=(num_users, 100)).astype(np.float32),
        "semantic": np.random.normal(semantic_mean, 0.35, size=(num_users, 50)).astype(np.float32),
        "syntax": np.random.normal(syntax_mean, 0.5, size=(num_users, 50)).astype(np.float32),
    }


def build_compare_features(num_users: int, member: bool) -> np.ndarray:
    """Build synthetic COMPARE 12D features: target NDCG plus 11 discrepancies."""
    target_mean = 0.72 if member else 0.45
    diff_mean = 0.16 if member else -0.03
    target_ndcg = np.random.normal(target_mean, 0.06, size=(num_users, 1))
    discrepancies = np.random.normal(diff_mean, 0.05, size=(num_users, 11))
    features = np.concatenate([target_ndcg, discrepancies], axis=1)
    return np.clip(features, -1.0, 1.0).astype(np.float32)


def print_summary(name: str, output) -> None:
    print(f"\n{name}")
    print("=" * len(name))
    print("first 5 scores:", np.asarray(output.membership_scores[:5]).round(4).tolist())
    print("first 5 preds: ", np.asarray(output.membership_preds[:5]).tolist())
    if output.evaluation is not None:
        print("accuracy:     ", round(output.evaluation.accuracy or 0.0, 4))
        print("auroc:        ", round(output.evaluation.auroc or 0.0, 4))
        print("tpr@fpr:      ", output.evaluation.tpr_at_fpr)


def run_me_mia_demo() -> None:
    feature_dim = 80
    shadow_member_scores = build_me_mia_scores(0, 48, feature_dim, member=True)
    shadow_nonmember_scores = build_me_mia_scores(1000, 48, feature_dim, member=False)
    target_member_scores = build_me_mia_scores(2000, 24, feature_dim, member=True)
    target_nonmember_scores = build_me_mia_scores(3000, 24, feature_dim, member=False)

    attack = MEMIAAttack()
    attack_input = AttackInput(
        target_model=None,
        samples={
            "member_scores": target_member_scores,
            "nonmember_scores": target_nonmember_scores,
        },
        membership_labels=np.concatenate(
            [
                np.ones(len(target_member_scores), dtype=np.int64),
                np.zeros(len(target_nonmember_scores), dtype=np.int64),
            ],
            axis=0,
        ),
        shadow_data={
            "member_scores": shadow_member_scores,
            "nonmember_scores": shadow_nonmember_scores,
        },
        config={
            "mia_data_mode": "mean",
            "classifier_mode": "mean",
            "batch_size": 32,
            "epochs": 60,
            "lr": 1e-3,
            "feature_start": 64,
            "val_check": True,
        },
    )

    output = attack.run(attack_input)
    print_summary("ME-MIA Demo", output)


def run_biased_mia_demo() -> None:
    item_embeddings = build_item_embeddings(200, 100)
    shadow_member_interactions, shadow_member_recommendations = build_raw_rec_payload(
        0, 96, item_embeddings, member=True
    )
    shadow_nonmember_interactions, shadow_nonmember_recommendations = build_raw_rec_payload(
        1000, 96, item_embeddings, member=False
    )
    target_member_interactions, target_member_recommendations = build_raw_rec_payload(
        2000, 40, item_embeddings, member=True
    )
    target_nonmember_interactions, target_nonmember_recommendations = build_raw_rec_payload(
        3000, 40, item_embeddings, member=False
    )

    attack = BiasedMIAAttack(
        hidden_dims=(32, 8),
        batch_size=32,
        lr=1e-2,
        momentum=0.7,
        epochs=20,
    )
    attack_input = AttackInput(
        target_model=None,
        samples={
            "member_interactions": target_member_interactions,
            "nonmember_interactions": target_nonmember_interactions,
            "member_recommendations": target_member_recommendations,
            "nonmember_recommendations": target_nonmember_recommendations,
            "item_embeddings": item_embeddings,
        },
        membership_labels=np.concatenate(
            [
                np.ones(len(target_member_interactions), dtype=np.int64),
                np.zeros(len(target_nonmember_interactions), dtype=np.int64),
            ],
            axis=0,
        ),
        shadow_data={
            "member_interactions": shadow_member_interactions,
            "nonmember_interactions": shadow_nonmember_interactions,
            "member_recommendations": shadow_member_recommendations,
            "nonmember_recommendations": shadow_nonmember_recommendations,
            "item_embeddings": item_embeddings,
        },
    )

    output = attack.run(attack_input)
    print_summary("Biased-MIA Demo", output)


def run_dl_mia_demo() -> None:
    item_embeddings = build_item_embeddings(200, 100)
    shadow_member_interactions, shadow_member_recommendations = build_raw_rec_payload(
        0, 96, item_embeddings, member=True
    )
    shadow_nonmember_interactions, shadow_nonmember_recommendations = build_raw_rec_payload(
        1000, 96, item_embeddings, member=False
    )
    target_member_interactions, target_member_recommendations = build_raw_rec_payload(
        2000, 40, item_embeddings, member=True
    )
    target_nonmember_interactions, target_nonmember_recommendations = build_raw_rec_payload(
        3000, 40, item_embeddings, member=False
    )

    attack = DLMIAAttack(
        hidden_dims=(32, 8),
        batch_size=32,
        lr=1e-3,
        epochs=40,
        alpha=0.1,
        beta=0.0,
    )
    attack_input = AttackInput(
        target_model=None,
        samples={
            "member_interactions": target_member_interactions,
            "nonmember_interactions": target_nonmember_interactions,
            "member_recommendations": target_member_recommendations,
            "nonmember_recommendations": target_nonmember_recommendations,
            "item_embeddings": item_embeddings,
        },
        membership_labels=np.concatenate(
            [
                np.ones(len(target_member_interactions), dtype=np.int64),
                np.zeros(len(target_nonmember_interactions), dtype=np.int64),
            ],
            axis=0,
        ),
        shadow_data={
            "member_interactions": shadow_member_interactions,
            "nonmember_interactions": shadow_nonmember_interactions,
            "member_recommendations": shadow_member_recommendations,
            "nonmember_recommendations": shadow_nonmember_recommendations,
            "item_embeddings": item_embeddings,
        },
    )

    output = attack.run(attack_input)
    print_summary("DL-MIA Demo", output)


def run_compare_mia_demo() -> None:
    shadow_member_features = build_compare_features(96, member=True)
    shadow_nonmember_features = build_compare_features(96, member=False)
    target_member_features = build_compare_features(40, member=True)
    target_nonmember_features = build_compare_features(40, member=False)

    attack = CompareMIAAttack(classifier="logistic_regression")
    attack_input = AttackInput(
        target_model=None,
        samples={
            "member_features": target_member_features,
            "nonmember_features": target_nonmember_features,
        },
        membership_labels=np.concatenate(
            [
                np.ones(len(target_member_features), dtype=np.int64),
                np.zeros(len(target_nonmember_features), dtype=np.int64),
            ],
            axis=0,
        ),
        shadow_data={
            "member_features": shadow_member_features,
            "nonmember_features": shadow_nonmember_features,
        },
    )

    output = attack.run(attack_input)
    print_summary("COMPARE Demo", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified recommender MIA demo")
    parser.add_argument(
        "--method",
        type=str,
        default="all",
        choices=["all", "me", "biased", "dl", "compare"],
        help="Which attack demo to run",
    )
    args = parser.parse_args()

    set_seed(2026)

    if args.method in ("all", "me"):
        run_me_mia_demo()
    if args.method in ("all", "biased"):
        run_biased_mia_demo()
    if args.method in ("all", "dl"):
        run_dl_mia_demo()
    if args.method in ("all", "compare"):
        run_compare_mia_demo()


if __name__ == "__main__":
    main()
