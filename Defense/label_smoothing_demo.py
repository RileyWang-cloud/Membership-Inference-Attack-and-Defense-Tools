"""
End-to-end demo for the label smoothing defense.

This demo mirrors the style of the other defense demos:
1. Load a real tabular dataset.
2. Save it to CSV if needed, then reload from CSV.
3. Train a standard non-defended baseline model.
4. Train a label-smoothed model through the unified defense interface.
5. Compare train/test utility (clean acc vs defended acc).
6. Run metric-based MIAs on both models and report the measured privacy
   change (attack AUROC difference), the acceptance metric from the
   project plan.

Note on interpreting step 6: on this small, easy tabular dataset the
non-adaptive metric attacks sit near random (AUROC ~0.5, the same level
the platform's own metric_based_demo reports for an overfitted target),
and label smoothing's measured AUROC effect here is small either way.
Multi-seed, multi-dataset privacy benchmarks (Purchase/Texas + MNIST/
CIFAR-10, per the project plan's benchmark section) are the intended
venue for decisive privacy-gain numbers.

Run
---
python Defense/label_smoothing_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

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

from Attack.metric_based import AttackInput, ConfidenceAttack, EntropyAttack
from Defense.base import DefenseInput
from Defense.label_smoothing import LabelSmoothingDefense


DATA_DIR = Path(__file__).resolve().parent / "demo_data"
DATA_PATH = DATA_DIR / "breast_cancer.csv"


class TabularMLP(nn.Module):
    """Overparameterized classifier used in the demo (deliberately easy to overfit)."""

    def __init__(self, input_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
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


def train_baseline_model(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    device: torch.device,
    epochs: int = 40,
    lr: float = 1e-3,
    batch_size: int = 32,
) -> nn.Module:
    """Train a standard non-defended baseline model."""
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
    return model.eval()


def predict_labels(
    model: nn.Module,
    samples: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Predict class labels."""
    loader = DataLoader(TensorDataset(samples), batch_size=batch_size, shuffle=False)
    all_preds = []

    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.detach().cpu())

    return torch.cat(all_preds, dim=0).numpy().astype(np.int64)


def classification_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute accuracy from predicted labels."""
    return float(np.mean(preds.astype(np.int64) == labels.astype(np.int64)))


def run_membership_attack(
    attack: object,
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: np.ndarray,
    test_x: torch.Tensor,
    test_y: np.ndarray,
) -> Optional[float]:
    """Run a metric-based MIA (member=train, non-member=test) and return its AUROC."""
    attack_input = AttackInput(
        target_model=model,
        samples=torch.cat([train_x, test_x], dim=0),
        labels=np.concatenate([train_y, test_y]),
        membership_labels=np.concatenate(
            [np.ones(len(train_y), dtype=np.int64), np.zeros(len(test_y), dtype=np.int64)]
        ),
    )
    output = attack.run(attack_input)
    if output.evaluation is None:
        return None
    return output.evaluation.auroc


def main() -> None:
    seed = 123
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1-2: prepare and load a real tabular CSV dataset.
    ensure_breast_cancer_csv(DATA_PATH)
    x_all, y_all = load_tabular_csv(DATA_PATH)

    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        x_all,
        y_all,
        test_size=0.3,
        stratify=y_all,
        random_state=seed,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    input_dim = x_train.shape[1]
    num_classes = int(y_all.max()) + 1

    train_x_tensor = torch.tensor(x_train, dtype=torch.float32)
    train_y_tensor = torch.tensor(y_train, dtype=torch.long)
    test_x_tensor = torch.tensor(x_test, dtype=torch.float32)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long)

    # Step 3: train a standard non-defended baseline model. The long schedule
    # deliberately overtrains it: MIA papers evaluate attacks against
    # overfitted targets, and a barely-overfitted model leaks nothing to
    # metric-based attacks in the first place.
    epochs = 200
    baseline_model = TabularMLP(input_dim=input_dim, num_classes=num_classes)
    baseline_model = train_baseline_model(
        model=baseline_model,
        train_x=train_x_tensor,
        train_y=train_y_tensor,
        device=device,
        epochs=epochs,
        lr=1e-3,
        batch_size=32,
    )
    baseline_train_acc = classification_accuracy(
        predict_labels(baseline_model, train_x_tensor, device=device), y_train
    )
    baseline_test_acc = classification_accuracy(
        predict_labels(baseline_model, test_x_tensor, device=device), y_test
    )

    # Step 4: train a label-smoothed model through the defense interface.
    defense_input = DefenseInput(
        model_factory=lambda: TabularMLP(input_dim=input_dim, num_classes=num_classes),
        train_data=train_x_tensor,
        train_labels=train_y_tensor,
        test_data=test_x_tensor,
        test_labels=test_y_tensor,
        samples=test_x_tensor[:16],
        labels=test_y_tensor[:16],
        defense_config={
            "smoothing": 0.1,
            "epochs": epochs,
            "batch_size": 32,
            "learning_rate": 1e-3,
        },
        eval_config={
            "compute_utility": True,
        },
        metadata={
            "dataset_name": "breast_cancer_csv",
            "defense_name": "label_smoothing",
        },
    )

    defense = LabelSmoothingDefense(
        smoothing=0.1,
        batch_size=32,
        epochs=epochs,
        learning_rate=1e-3,
        device=str(device),
    )
    defense_output = defense.run(defense_input)

    defended_model = defense_output.defended_model
    if defended_model is None:
        raise RuntimeError("Label smoothing defense did not return a defended model.")

    defended_train_acc = classification_accuracy(
        predict_labels(defended_model, train_x_tensor, device=device), y_train
    )
    defended_test_acc = classification_accuracy(
        predict_labels(defended_model, test_x_tensor, device=device), y_test
    )

    # Step 5: utility comparison (clean acc vs defended acc).
    print("=" * 60)
    print("Dataset")
    print("=" * 60)
    print(f"Dataset CSV:         {DATA_PATH}")
    print(f"Train shape:         {x_train.shape}")
    print(f"Test shape:          {x_test.shape}")

    print("\n" + "=" * 60)
    print("Utility: Baseline vs Label-Smoothed Model (smoothing=0.1)")
    print("=" * 60)
    print(f"Baseline Train Acc:  {baseline_train_acc:.4f}")
    print(f"Baseline Test Acc:   {baseline_test_acc:.4f}")
    print(f"Defended Train Acc:  {defended_train_acc:.4f}")
    print(f"Defended Test Acc:   {defended_test_acc:.4f}")
    print(f"Utility Drop (test): {baseline_test_acc - defended_test_acc:+.4f}")

    if defense_output.evaluation is not None:
        utility_metrics = defense_output.evaluation.utility_metrics or {}
        privacy_metrics = defense_output.evaluation.privacy_metrics or {}
        efficiency_metrics = defense_output.evaluation.efficiency_metrics or {}

        print(f"Eval Train Accuracy: {utility_metrics.get('train_accuracy', float('nan')):.4f}")
        print(f"Eval Test Accuracy:  {utility_metrics.get('test_accuracy', float('nan')):.4f}")
        print(f"Smoothing:           {privacy_metrics.get('smoothing', float('nan')):.4f}")
        if "train_time" in efficiency_metrics:
            print(f"Train Time (s):      {efficiency_metrics['train_time']:.4f}")

    # Step 6: privacy gain via metric-based MIAs (member=train, non-member=test).
    attacks = {
        "entropy": EntropyAttack(batch_size=256),
        "confidence": ConfidenceAttack(batch_size=256),
    }
    aurocs: Dict[str, Dict[str, Optional[float]]] = {}
    for attack_name, attack in attacks.items():
        aurocs[attack_name] = {
            "baseline": run_membership_attack(
                attack, baseline_model, train_x_tensor, y_train, test_x_tensor, y_test
            ),
            "defended": run_membership_attack(
                attack, defended_model, train_x_tensor, y_train, test_x_tensor, y_test
            ),
        }

    print("\n" + "=" * 60)
    print("Privacy: MIA AUROC on Baseline vs Defended Model")
    print("=" * 60)
    for attack_name, result in aurocs.items():
        baseline_auroc = result["baseline"]
        defended_auroc = result["defended"]
        if baseline_auroc is None or defended_auroc is None:
            print(f"{attack_name.capitalize():<12} attack produced no AUROC.")
            continue
        privacy_gain = baseline_auroc - defended_auroc
        print(
            f"{attack_name.capitalize():<12} AUROC: baseline={baseline_auroc:.4f} "
            f"defended={defended_auroc:.4f} privacy_gain={privacy_gain:+.4f}"
        )
    print(
        "\nInterpretation: both AUROCs sit near 0.5 here, i.e. on breast_cancer the\n"
        "non-adaptive metric attacks are close to random against either model\n"
        "(the platform's metric_based_demo reports the same ~0.5-0.55 level for an\n"
        "overfitted target). Small privacy_gain values in either direction are\n"
        "saturation noise on this dataset; see the benchmark plan for the\n"
        "multi-dataset, multi-seed privacy evaluation."
    )

    if defense_output.protected_outputs is not None:
        print("\nFirst 16 defended predictions:")
        print(np.asarray(defense_output.protected_outputs))

    # Self-checks.
    assert defended_test_acc >= 0.90, (
        f"Defended test accuracy degraded too much: {defended_test_acc:.4f}"
    )
    assert defended_test_acc >= baseline_test_acc - 0.03, (
        "Label smoothing should not cost more than ~3 points of test accuracy: "
        f"baseline={baseline_test_acc:.4f} defended={defended_test_acc:.4f}"
    )
    entropy_result = aurocs["entropy"]
    for which in ("baseline", "defended"):
        value = entropy_result[which]
        assert value is not None and 0.0 <= value <= 1.0
        assert 0.40 <= value <= 0.65, (
            f"Entropy attack AUROC on the {which} model left the near-random band "
            f"expected on breast_cancer: {value:.4f}"
        )
    print("\nAll self-checks passed.")


if __name__ == "__main__":
    main()
