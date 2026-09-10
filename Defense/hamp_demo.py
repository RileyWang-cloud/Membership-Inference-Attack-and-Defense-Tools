"""Minimal end-to-end HAMP example.

Run:
    python Defense/hamp_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Defense.base import DefenseInput
from Defense.hamp import HAMPDefense


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
    generator = torch.Generator().manual_seed(17)
    weights = torch.randn(8, 3, generator=generator)

    def sample(count: int) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.randn(count, 8, generator=generator)
        labels = (features @ weights).argmax(dim=1)
        return features, labels

    train_x, train_y = sample(160)
    val_x, val_y = sample(64)
    test_x, test_y = sample(80)
    return train_x, train_y, val_x, val_y, test_x, test_y


def main() -> None:
    torch.manual_seed(17)
    train_x, train_y, val_x, val_y, test_x, test_y = make_data()
    output = HAMPDefense(device="cpu").run(
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
                "epochs": 5,
                "batch_size": 32,
                "optimizer": "adam",
                "learning_rate": 0.01,
                "entropy_percentile": 0.8,
                "entropy_penalty": True,
                "entropy_weight": 0.01,
                "random_input_mode": "normal",
            },
            eval_config={"enabled": True},
        )
    )

    raw_logits = np.asarray(output.intermediate_outputs["raw_logits"])
    protected_logits = np.asarray(output.protected_outputs)
    assert output.defended_model is not None
    assert output.protected_predictor is not None
    assert np.array_equal(raw_logits.argmax(axis=1), protected_logits.argmax(axis=1))
    assert np.array_equal(np.argsort(raw_logits, axis=1), np.argsort(protected_logits, axis=1))
    print("HAMP output metadata:", output.metadata)
    print("Utility metrics:", output.evaluation.utility_metrics)
    print("Privacy metrics:", output.evaluation.privacy_metrics)
    print("All HAMP self-checks passed.")


if __name__ == "__main__":
    main()
