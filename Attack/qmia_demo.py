"""
End-to-end demo for the QMIA attack using the project's minimal attack
interface.

Pipeline (mirrors the other attack demos):
1. Build a synthetic dataset of random Gaussian features with RANDOM labels.
   The target model can only fit this by memorising training points — which is
   exactly the memorisation vulnerability MIA (and QMIA) exploits. This is the
   canonical setup for demonstrating membership inference.
2. Train a target MLP on the member pool (it memorises them: ~100% train acc,
   chance test acc).
3. Train QMIA's quantile-regression head on a held-out offline (non-member)
   pool via augmentation + pinball loss.
4. Query a mix of TRUE members (a subset of the training pool) and non-members
   (held out).
5. Run QMIA via ``QMIAAttack.run`` and report the unified evaluation.

Run
---
python Attack/qmia_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.base import AttackInput
from Attack.qmia import QMIAAttack


SEED = 0
NUM_CLASSES = 5
NUM_FEATURES = 32
N_TOTAL = 1600              # split 50/50 into member pool and non-member pool
N_QUERY_PER_SIDE = 200      # members (subset of train) + non-members to attack
TARGET_EPOCHS = 120
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data():
    rng = np.random.default_rng(SEED)
    # Random Gaussian features with RANDOM labels => the model must memorise.
    X = rng.normal(size=(N_TOTAL, NUM_FEATURES)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=N_TOTAL).astype(np.int64)

    X_mem, X_nonmem, y_mem, y_nonmem = train_test_split(
        X, y, test_size=0.5, random_state=SEED, stratify=y
    )
    # Member queries are TRUE members: sampled from (and kept inside) the train pool.
    q_idx = rng.choice(len(X_mem), size=N_QUERY_PER_SIDE, replace=False)
    X_query_mem, y_query_mem = X_mem[q_idx], y_mem[q_idx]
    # Offline data (trains the quantile head) + non-member queries: held-out pool.
    X_offline, X_query_nonmem, y_offline, y_query_nonmem = train_test_split(
        X_nonmem, y_nonmem, test_size=N_QUERY_PER_SIDE, random_state=SEED, stratify=y_nonmem
    )
    return rng, (X_mem, y_mem), (X_offline, y_offline), \
        (X_query_mem, y_query_mem), (X_query_nonmem, y_query_nonmem)


def train_target(X_train, y_train) -> TabularMLP:
    model = TabularMLP(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64,
        shuffle=True,
    )
    model.train()
    for _ in range(TARGET_EPOCHS):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    return model


def accuracy(model, X, y) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(DEVICE))
        pred = logits.argmax(1).cpu().numpy()
    return float((pred == y).mean())


def main() -> None:
    torch.manual_seed(SEED)
    rng, (Xt, yt), (Xoff, yoff), (Xqm, yqm), (Xqn, yqn) = load_data()

    target = train_target(Xt, yt)
    print(f"target train accuracy (members): {accuracy(target, Xt, yt):.3f}")
    print(f"target non-member accuracy      : {accuracy(target, Xqn, yqn):.3f}")

    # Query set: TRUE members first, then non-members; labels aligned.
    X_query = np.concatenate([Xqm, Xqn], axis=0)
    y_query = np.concatenate([yqm, yqn], axis=0)
    membership = np.concatenate([np.ones(len(Xqm)), np.zeros(len(Xqn))]).astype(np.int64)

    attack_input = AttackInput(
        target_model=target,
        samples=X_query,
        labels=y_query,
        shadow_data={"fit_X": Xoff, "fit_y": yoff},
        membership_labels=membership,
        config={"operating_quantile": 0.9, "n_augmentations": 16, "aug_noise": 0.3, "n_epochs": 50},
    )

    output = QMIAAttack().run(attack_input)

    scores = np.asarray(output.membership_scores, dtype=np.float64)
    preds = np.asarray(output.membership_preds, dtype=np.int64)
    members = membership == 1

    print("=" * 64)
    print("QMIA demo")
    print("=" * 64)
    print(f"queries: {len(X_query)} (members={int(members.sum())}, "
          f"non-members={int((~members).sum())})")
    print(f"offline (quantile-head training): {len(Xoff)}")
    print(f"operating quantile used: {output.intermediate_outputs['operating_quantile']:.4f}")
    print(f"mean score  members    : {scores[members].mean():+.4f}")
    print(f"mean score  non-members: {scores[~members].mean():+.4f}")
    print(f"pred accuracy (vs labels): {(preds == membership).mean():.3f}")
    print("-" * 64)
    print(f"evaluation: {output.evaluation}")
    print("=" * 64)

    # Self-checks (design doc §7.1, §10.3: higher score -> more likely member).
    assert scores[members].mean() > scores[~members].mean(), (
        "member mean score should exceed non-member mean score"
    )
    assert output.evaluation is not None and output.evaluation.auroc is not None
    assert output.evaluation.auroc > 0.9, (
        f"AUROC expected > 0.9 on the memorised scenario, got {output.evaluation.auroc:.3f}"
    )
    # Hard decisions use the offline reference-calibrated threshold, so accuracy
    # should be well above chance.
    assert (preds == membership).mean() > 0.7, (
        f"pred accuracy expected > 0.7, got {(preds == membership).mean():.3f}"
    )
    print("All self-checks passed.")


if __name__ == "__main__":
    main()
