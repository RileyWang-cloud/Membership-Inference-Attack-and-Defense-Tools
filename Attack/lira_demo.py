"""
End-to-end demo for LiRA using the project's minimal attack interface.

This demo mirrors the style of `shadow_based_demo.py`:
1. Load a real tabular dataset.
2. Save it as CSV if needed, then reload from CSV.
3. Train a target MLP classifier.
4. Build LiRA reference-model inputs from train/test pools.
5. Attack a mixture of member and non-member samples.
6. Evaluate LiRA attack performance.

Run
---
python Attack/lira_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.base import AttackInput
from Attack.lira import LiRAAttack


DATA_DIR = Path(__file__).resolve().parent / "demo_data"
DATA_PATH = DATA_DIR / "breast_cancer.csv"


class TabularMLP(nn.Module):
    """Small classifier used for both target and reference models."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def ensure_breast_cancer_csv(path: Path) -> None:
    """Export the sklearn breast cancer dataset to CSV once."""
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    df = dataset.frame.copy().rename(columns={"target": "label"})
    df.to_csv(path, index=False)


def load_tabular_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load features and labels from a CSV file with a `label` column."""
    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("CSV file must contain a 'label' column.")
    x = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return x, y


def train_classifier(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 60,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> nn.Module:
    """Train a small tabular classifier."""
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    return model


def classification_accuracy(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute target-model accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        preds = logits.argmax(dim=1).cpu()
    return float((preds == y.cpu()).float().mean().item())


def main() -> None:
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1-3: prepare and load a real tabular CSV dataset.
    ensure_breast_cancer_csv(DATA_PATH)
    x_all, y_all = load_tabular_csv(DATA_PATH)

    # Split into the target model's member/non-member pools.
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.5,
        stratify=y_all,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    input_dim = x_train.shape[1]
    num_classes = int(y_all.max()) + 1

    # Step 4: train the target model.
    target_model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    target_model = train_classifier(
        model=target_model,
        train_x=torch.tensor(x_train, dtype=torch.float32),
        train_y=torch.tensor(y_train, dtype=torch.long),
        device=device,
        epochs=80,
        lr=1e-3,
        batch_size=64,
    )

    target_train_acc = classification_accuracy(
        target_model,
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
        device,
    )
    target_test_acc = classification_accuracy(
        target_model,
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
        device,
    )

    # Step 5: assemble attacked samples.
    # Members: target train samples.
    # Non-members: balanced subset from target test samples.
    rng = np.random.default_rng(seed + 1)
    nonmember_idx = rng.choice(len(x_test), size=min(len(x_train), len(x_test)), replace=False)
    x_test_attack = x_test[nonmember_idx]
    y_test_attack = y_test[nonmember_idx]

    attack_samples = np.concatenate([x_train, x_test_attack], axis=0)
    attack_labels = np.concatenate([y_train, y_test_attack], axis=0)
    membership_labels = np.concatenate(
        [
            np.ones(len(x_train), dtype=np.int64),
            np.zeros(len(x_test_attack), dtype=np.int64),
        ],
        axis=0,
    )

    # LiRA needs sample indices aligned with the reference train/test pools.
    sample_indices = np.concatenate(
        [
            np.arange(len(x_train), dtype=np.int64),
            nonmember_idx.astype(np.int64) + len(x_train),
        ],
        axis=0,
    )

    # Step 6: define attack input and run LiRA.
    attack_input = AttackInput(
        target_model=target_model,
        samples=torch.tensor(attack_samples, dtype=torch.float32),
        labels=torch.tensor(attack_labels, dtype=torch.long),
        membership_labels=torch.tensor(membership_labels, dtype=torch.long),
        reference_data={
            "train_X": x_train,
            "train_y": y_train,
            "test_X": x_test,
            "test_y": y_test,
            "model_factory": lambda: TabularMLP(input_dim=input_dim, num_classes=num_classes),
            "train_config": {
                "epochs": 40,
                "lr": 1e-3,
            },
            "data_sizes": [64],
            "random_seed_num": 3,
            "reference_model_number": 6,
            "batch_size": 64,
        },
        config={"batch_size": 64},
        metadata={
            "dataset_name": "breast_cancer_csv",
            "attack_name": "lira",
            "sample_indices": sample_indices,
        },
    )

    attack = LiRAAttack(
        data_sizes=[64],
        random_seed_num=3,
        reference_model_number=6,
        batch_size=64,
        device=str(device),
    )
    attack_output = attack.run(attack_input)

    # Step 7: print results.
    print("=" * 60)
    print("Target Model Performance")
    print("=" * 60)
    print(f"Dataset CSV:        {DATA_PATH}")
    print(f"Train Accuracy:     {target_train_acc:.4f}")
    print(f"Test Accuracy:      {target_test_acc:.4f}")

    print("\n" + "=" * 60)
    print("LiRA Attack Performance")
    print("=" * 60)
    print(f"Attacked samples:   {len(attack_samples)}")
    print(f"Members:            {int(membership_labels.sum())}")
    print(f"Non-members:        {int((1 - membership_labels).sum())}")

    if attack_output.evaluation is not None:
        eval_result = attack_output.evaluation
        print(f"Attack Accuracy:    {eval_result.accuracy:.4f}")
        print(f"Attack AUROC:       {eval_result.auroc:.4f}")
        print(f"TPR@1%FPR:          {eval_result.tpr_at_fpr['1%']:.4f}")
        print(f"TPR@0.1%FPR:        {eval_result.tpr_at_fpr['0.1%']:.4f}")

    if attack_output.intermediate_outputs is not None:
        print(
            f"Reference Coverage: "
            f"{attack_output.intermediate_outputs['reference_coverage']:.4f}"
        )

    print("\nFirst 10 LiRA scores:")
    print(np.asarray(attack_output.membership_scores[:10]))


if __name__ == "__main__":
    main()
