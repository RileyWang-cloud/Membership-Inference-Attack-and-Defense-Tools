"""Minimal end-to-end adversarial-regularization example.

Run:
    python Defense/adv_reg_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Defense.adv_reg import AdvRegDefense
from Defense.base import DefenseInput


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.ReLU(),
            nn.Linear(24, num_classes),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(samples)


def make_data() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(29)
    weights = torch.randn(8, 3, generator=generator)

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.randn(count, 8, generator=generator)
        labels = (features @ weights).argmax(dim=1)
        return features, labels

    train_x, train_y = sample(128)
    pseudo_nonmember_x, pseudo_nonmember_y = sample(128)
    test_x, test_y = sample(80)
    return train_x, train_y, pseudo_nonmember_x, pseudo_nonmember_y, test_x, test_y


def main() -> None:
    torch.manual_seed(29)
    train_x, train_y, nonmember_x, nonmember_y, test_x, test_y = make_data()
    output = AdvRegDefense(device="cpu").run(
        DefenseInput(
            model_factory=TinyClassifier,
            train_data=train_x,
            train_labels=train_y,
            test_data=test_x,
            test_labels=test_y,
            samples=test_x[:12],
            labels=test_y[:12],
            auxiliary_data={
                "nonmember_data": nonmember_x,
                "nonmember_labels": nonmember_y,
            },
            defense_config={
                "epochs": 4,
                "warmup_epochs": 1,
                "adversary_steps": 1,
                "batch_size": 32,
                "learning_rate": 0.01,
                "adversary_learning_rate": 0.005,
                "alpha": 0.5,
            },
            eval_config={"enabled": True},
        )
    )

    assert output.defended_model is not None
    assert output.protected_outputs.shape == (12,)
    assert output.artifacts["membership_adversary"] is not None
    assert len(output.artifacts["training_history"]) == 4
    print("AdvReg output metadata:", output.metadata)
    print("Utility metrics:", output.evaluation.utility_metrics)
    print("Privacy metrics:", output.evaluation.privacy_metrics)
    print("All AdvReg self-checks passed.")


if __name__ == "__main__":
    main()
