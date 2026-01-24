"""
Shadow-Free Membership Inference Attack (SFMD) - IJCAI-24

Core algorithm: S1 > S2 -> Member, else Non-member
- S1: similarity(interaction_vector, recommendation_vector)
- S2: similarity(recommendation_vector, baseline_vector)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple


class ShadowFreeMIA:
    """Shadow-Free Membership Inference Attack - no shadow training needed."""

    def __init__(self, item_embeddings: Dict[int, torch.Tensor], num_latent: int = 100):
        self.item_embeddings = item_embeddings
        self.num_latent = num_latent
        self.baseline_vector = None

    def compute_baseline_vector(self, popular_items: List[int], top_k: int = 100) -> torch.Tensor:
        """Compute baseline vector from top-k popular items."""
        temp_vector = torch.zeros(self.num_latent)
        count = 0
        i = 0

        while count < top_k and i < len(popular_items):
            item_id = popular_items[i]
            if item_id in self.item_embeddings:
                temp_vector = temp_vector + self.item_embeddings[item_id]
                count += 1
            i += 1

        self.baseline_vector = temp_vector / count if count > 0 else torch.zeros(self.num_latent)
        return self.baseline_vector

    def compute_user_vector(self, items: List[int]) -> torch.Tensor:
        """Compute user vector from item list."""
        temp_vector = torch.zeros(self.num_latent)
        valid_count = 0

        for item_id in items:
            if item_id in self.item_embeddings:
                temp_vector = temp_vector + self.item_embeddings[item_id]
                valid_count += 1

        return temp_vector / valid_count if valid_count > 0 else torch.zeros(self.num_latent)

    def compute_inverse_euclidean_distance(self, vec1: torch.Tensor, vec2: torch.Tensor) -> float:
        """Compute inverse Euclidean distance as similarity metric."""
        diff = torch.subtract(vec1, vec2)
        distance = torch.sqrt(torch.sum(torch.pow(diff, 2), dim=0))
        return (1.0 / (distance + 1e-8)).item()

    def attack_single_user(
        self,
        interaction_items: List[int],
        recommended_items: List[int]
    ) -> Tuple[int, float, float]:
        """Attack single user. Returns (prediction, S1, S2)."""
        if self.baseline_vector is None:
            raise ValueError("Baseline vector not computed. Call compute_baseline_vector() first.")

        interaction_vector = self.compute_user_vector(interaction_items)
        recommendation_vector = self.compute_user_vector(recommended_items)

        S1 = self.compute_inverse_euclidean_distance(interaction_vector, recommendation_vector)
        S2 = self.compute_inverse_euclidean_distance(recommendation_vector, self.baseline_vector)

        prediction = 1 if S1 > S2 else 0
        return prediction, S1, S2

    def attack_multiple_users(
        self,
        interaction_data: Dict[int, List[int]],
        recommendation_data: Dict[int, List[int]],
        true_labels: Dict[int, int] = None
    ) -> Dict:
        """Attack multiple users and compute metrics if labels provided."""
        results = {'predictions': {}, 'S1_scores': {}, 'S2_scores': {}, 'metrics': {}}

        member_count = nonmem_count = total_members = total_nonmem = 0

        for user_id in interaction_data:
            if user_id not in recommendation_data:
                continue

            prediction, S1, S2 = self.attack_single_user(
                interaction_data[user_id],
                recommendation_data[user_id]
            )

            results['predictions'][user_id] = prediction
            results['S1_scores'][user_id] = S1
            results['S2_scores'][user_id] = S2

            if true_labels and user_id in true_labels:
                true_label = true_labels[user_id]
                if true_label == 1:
                    total_members += 1
                    if prediction == 1:
                        member_count += 1
                else:
                    total_nonmem += 1
                    if prediction == 0:
                        nonmem_count += 1

        if true_labels:
            results['metrics'] = {
                'attack_success_rate': (member_count + nonmem_count) / (total_members + total_nonmem),
                'true_positive_rate': member_count / total_members if total_members > 0 else 0,
                'false_positive_rate': (total_nonmem - nonmem_count) / total_nonmem if total_nonmem > 0 else 0,
                'members_correct': member_count,
                'total_members': total_members,
                'nonmembers_correct': nonmem_count,
                'total_nonmembers': total_nonmem
            }

        return results

    def get_similarity_scores(
        self,
        interaction_data: Dict[int, List[int]],
        recommendation_data: Dict[int, List[int]]
    ) -> Tuple[List[float], List[float], List[float]]:
        """Get similarity scores (S1, S2, S1-S2) for all users."""
        S1_scores, S2_scores, S1_minus_S2 = [], [], []

        for user_id in interaction_data:
            if user_id not in recommendation_data:
                continue

            _, S1, S2 = self.attack_single_user(
                interaction_data[user_id],
                recommendation_data[user_id]
            )

            S1_scores.append(S1)
            S2_scores.append(S2)
            S1_minus_S2.append(S1 - S2)

        return S1_scores, S2_scores, S1_minus_S2


def example_usage():
    """Example usage of ShadowFreeMIA."""
    num_latent = 100
    item_embeddings = {i: torch.randn(num_latent) for i in range(1000)}

    attack_model = ShadowFreeMIA(item_embeddings, num_latent)
    attack_model.compute_baseline_vector(list(range(100)), top_k=100)

    interaction_data = {0: [1, 5, 10, 15, 20], 1: [100, 105, 110, 115]}
    recommendation_data = {0: [1, 5, 10, 25, 30], 1: [200, 205, 210, 215]}
    true_labels = {0: 1, 1: 0}

    results = attack_model.attack_multiple_users(interaction_data, recommendation_data, true_labels)

    print("\nAttack Results:")
    print(f"Predictions: {results['predictions']}")
    print(f"\nMetrics:")
    for metric, value in results['metrics'].items():
        print(f"  {metric}: {value:.4f}")

    S1_list, S2_list, S1_minus_S2 = attack_model.get_similarity_scores(interaction_data, recommendation_data)
    print(f"\nSimilarity Scores:")
    print(f"  S1: {S1_list}")
    print(f"  S2: {S2_list}")
    print(f"  S1 - S2: {S1_minus_S2}")


if __name__ == "__main__":
    example_usage()
