"""
End-to-end demo for the Enhanced MIA attack using the project's minimal attack
interface.

Setup (same memorisation trick as ``qmia_demo.py``): random Gaussian features
with RANDOM labels force the target classifier to memorise its training points.
Members then have low loss / high confidence, non-members high loss / low
confidence — rich signal for a learned attack model.

Pipeline:
1. Build random-label data; train a target MLP to memorise the member pool.
2. Fit the enhanced attack model on known member / non-member feature vectors
   (loss, correctness, confidence, entropy, margin, ...).
3. Query a mix of TRUE members (subset of train) and non-members.
4. Run via ``EnhancedMIAAttack.run`` and report the unified evaluation.

Run
---
python Attack/enhanced_mia_demo.py
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
from Attack.enhanced_mia import EnhancedMIAAttack


SEED = 0
NUM_CLASSES = 5
NUM_FEATURES = 32
N_TOTAL = 1600
N_QUERY_PER_SIDE = 200
TARGET_EPOCHS = 120
ATTACK_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_data():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_TOTAL, NUM_FEATURES)).astype(np.float32)
    y = rng.integers(0, NUM_CLASSES, size=N_TOTAL).astype(np.int64)
    X_mem, X_nonmem, y_mem, y_nonmem = train_test_split(X, y, test_size=0.5, random_state=SEED, stratify=y)
    # Target trains on the WHOLE member pool; shadow members and member queries
    # are sampled FROM it (TRUE members).
    mt_X, mt_y = X_mem, y_mem
    sm_idx = rng.choice(len(mt_X), size=300, replace=False)
    s_mem_X, s_mem_y = mt_X[sm_idx], mt_y[sm_idx]
    mq_idx = rng.choice(len(mt_X), size=N_QUERY_PER_SIDE, replace=False)
    mem_query_X, mem_query_y = mt_X[mq_idx], mt_y[mq_idx]
    # Non-members (shadow + queries) come from the held-out pool.
    s_non_X, non_query_X, s_non_y, non_query_y = train_test_split(
        X_nonmem, y_nonmem, test_size=N_QUERY_PER_SIDE, random_state=SEED, stratify=y_nonmem
    )
    return (mt_X, mt_y), (s_mem_X, s_mem_y, s_non_X, s_non_y), \
        (mem_query_X, mem_query_y), (non_query_X, non_query_y)


def train_target(X_train, y_train) -> TabularMLP:
    torch.manual_seed(SEED)
    model = TabularMLP(NUM_FEATURES, NUM_CLASSES).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=64, shuffle=True,
    )
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
        pred = model(torch.from_numpy(X).to(DEVICE)).argmax(1).cpu().numpy()
    return float((pred == y).mean())


def main() -> None:
    torch.manual_seed(SEED)
    (mt_X, mt_y), (s_mem_X, s_mem_y, s_non_X, s_non_y), (qm_X, qm_y), (qn_X, qn_y) = load_data()

    target = train_target(mt_X, mt_y)
    print(f"target train accuracy (members): {accuracy(target, mt_X, mt_y):.3f}")
    print(f"target non-member accuracy      : {accuracy(target, qn_X, qn_y):.3f}")

    X_query = np.concatenate([qm_X, qn_X])
    y_query = np.concatenate([qm_y, qn_y])
    membership = np.concatenate([np.ones(len(qm_X)), np.zeros(len(qn_X))]).astype(np.int64)

    attack_input = AttackInput(
        target_model=target,
        samples=X_query,
        labels=y_query,
        shadow_data={"member_X": s_mem_X, "member_y": s_mem_y,
                     "nonmember_X": s_non_X, "nonmember_y": s_non_y},
        membership_labels=membership,
        config={"epochs": ATTACK_EPOCHS, "use_label": True},
    )
    output = EnhancedMIAAttack().run(attack_input)

    scores = np.asarray(output.membership_scores, dtype=np.float64)
    preds = np.asarray(output.membership_preds, dtype=np.int64)
    members = membership == 1

    print("=" * 64)
    print("Enhanced MIA demo (NN attack on enhanced per-sample features)")
    print("=" * 64)
    print(f"queries: {len(X_query)} (members={int(members.sum())}, "
          f"non-members={int((~members).sum())})")
    print(f"shadow fit set: {len(s_mem_X)} members + {len(s_non_X)} non-members")
    print(f"features: {output.metadata['features']}  (use_label={output.metadata['use_label']})")
    print(f"mean member-prob  members    : {scores[members].mean():.4f}")
    print(f"mean member-prob  non-members: {scores[~members].mean():.4f}")
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
    print("All self-checks passed.")


if __name__ == "__main__":
    main()
