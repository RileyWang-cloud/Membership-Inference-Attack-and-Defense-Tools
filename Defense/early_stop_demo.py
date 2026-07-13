"""Minimal end-to-end EarlyStop example.

Run:
    python Defense/early_stop_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Defense.base import DefenseInput
from Defense.early_stop import EarlyStopDefense


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 8, num_classes: int = 3) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(samples)


def make_data() -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(23)
    weights = torch.randn(8, 3, generator=generator)

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.randn(count, 8, generator=generator)
        labels = (features @ weights).argmax(dim=1)
        return features, labels

    train_x, train_y = sample(96)
    val_x, val_y = sample(64)
    test_x, test_y = sample(80)
    return train_x, train_y, val_x, val_y, test_x, test_y


def main() -> None:
    torch.manual_seed(23)
    train_x, train_y, val_x, val_y, test_x, test_y = make_data()
    output = EarlyStopDefense(device="cpu").run(
        DefenseInput(
            model_factory=TinyClassifier,
            train_data=train_x,
            train_labels=train_y,
            val_data=val_x,
            val_labels=val_y,
            test_data=test_x,
            test_labels=test_y,
            samples=test_x[:12],
            labels=test_y[:12],
            defense_config={
                "epochs": 30,
                "batch_size": 32,
                "learning_rate": 0.02,
                "patience": 3,
                "min_delta": 0.002,
                "monitor": "val_loss",
            },
            eval_config={"enabled": True},
        )
    )

    assert output.defended_model is not None
    assert output.protected_outputs.shape == (12,)
    assert 1 <= output.metadata["selected_epoch"] <= output.metadata["stopped_epoch"] <= 30
    print("EarlyStop output metadata:", output.metadata)
    print("Utility metrics:", output.evaluation.utility_metrics)
    print("Privacy metrics:", output.evaluation.privacy_metrics)
    print("All EarlyStop self-checks passed.")


if __name__ == "__main__":
    main()
