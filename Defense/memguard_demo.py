"""Minimal end-to-end MemGuard example with a small attack benchmark.

Run:
    python Defense/memguard_demo.py

The demo trains an intentionally over-fitted classifier, wraps it with
``MemGuardDefense`` -- which first trains the defender's surrogate membership
classifier on member vs non-member posteriors and then perturbs every
released posterior with the official adversarial optimization (CCS 2019) --
and verifies that

1. utility survives: argmax predictions (and therefore accuracy) are unchanged,
2. the surrogate attack model can no longer separate members from non-members,
3. the toolkit's own metric attacks (loss / confidence / entropy /
   modified-entropy) lose most of their AUROC against the protected predictor.

The surrogate keeps the official architecture, features (sorted
posteriors), and loss (BCE); the documented deviations (Adam instead of the
official SGD constants, best-of-restarts surrogate training, and the
boundary-refinement stopping rules that small calibration sets need) are
all listed in Defense/memguard.py's module docstring.  Everything runs on
CPU; results are reported for seeds 0/1/2 as mean +/- std.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.metric_based import (
    AttackInput,
    ConfidenceAttack,
    CorrectnessAttack,
    EntropyAttack,
    LossAttack,
    ModifiedEntropyAttack,
)
from Defense._classification import predict_logits
from Defense.base import DefenseInput
from Defense.memguard import MemGuardDefense


class TinyClassifier(nn.Module):
    def __init__(self, input_dim: int = 16, num_classes: int = 10) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.network(samples)


def make_data(seed: int) -> Dict[str, torch.Tensor]:
    """Synthetic 10-class data whose labels come from a random teacher MLP.

    A fraction of the training labels is randomized so the target model has
    to memorize them, which creates a clear member/non-member gap.
    """
    generator = torch.Generator().manual_seed(seed)
    weights = torch.randn(16, 10, generator=generator)

    def sample(count: int):
        features = torch.randn(count, 16, generator=generator)
        labels = (features @ weights).argmax(dim=1)
        return features, labels

    train_x, train_y = sample(300)
    noise = torch.rand(len(train_y), generator=generator)
    train_y = torch.where(
        noise < 0.10, torch.randint(0, 10, (len(train_y),), generator=generator), train_y
    )
    reference_x, _reference_y = sample(400)  # non-member calibration pool
    test_x, test_y = sample(400)
    return {
        "train_x": train_x,
        "train_y": train_y,
        "reference_x": reference_x,
        "test_x": test_x,
        "test_y": test_y,
    }


def train_target_model(data: Dict[str, torch.Tensor], seed: int) -> TinyClassifier:
    torch.manual_seed(seed)
    model = TinyClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(250):
        logits = model(data["train_x"])
        loss = nn.functional.cross_entropy(logits, data["train_y"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def attack_aurocs(target, samples, labels, membership) -> Dict[str, float]:
    """Run the toolkit's metric attacks against `target` (model or predictor)."""
    results: Dict[str, float] = {}
    attacks = {
        "loss": LossAttack(),
        "correctness": CorrectnessAttack(),
        "confidence": ConfidenceAttack(),
        "entropy": EntropyAttack(),
        "modified_entropy": ModifiedEntropyAttack(),
    }
    for name, attack in attacks.items():
        output = attack.run(
            AttackInput(
                target_model=target,
                samples=samples,
                labels=labels,
                membership_labels=membership,
            )
        )
        results[name] = float(output.evaluation.auroc)
    return results


def run_seed(seed: int) -> Dict[str, float]:
    print(f"\n=== seed {seed} ===")
    data = make_data(seed)
    model = train_target_model(data, seed)

    with torch.no_grad():
        train_acc = float((model(data["train_x"]).argmax(dim=1) == data["train_y"]).float().mean())
        test_acc = float((model(data["test_x"]).argmax(dim=1) == data["test_y"]).float().mean())
    print(f"target model: train acc {train_acc:.3f} | test acc {test_acc:.3f}")

    # ---- MemGuard on the deployed model (inference-time, model unchanged) ----
    defense = MemGuardDefense(device="cpu")
    output = defense.run(
        DefenseInput(
            target_model=model,
            samples=data["test_x"][:16],
            labels=data["test_y"][:16],
            auxiliary_data={"nonmember_data": data["reference_x"]},
            train_data=data["train_x"],
            train_labels=data["train_y"],
            test_data=data["test_x"],
            test_labels=data["test_y"],
            eval_config={"enabled": True},
        )
    )
    evaluation = output.evaluation
    stats = output.artifacts["perturbation_stats"]
    print(
        "surrogate train accuracy:",
        round(output.artifacts["surrogate"]["train_accuracy"], 3),
        "| perturbed fraction:",
        round(stats["perturbed_fraction"], 3),
        "| surrogate |score-0.5|:",
        round(stats["surrogate_score_gap_before"], 3),
        "->",
        round(stats["surrogate_score_gap_after"], 3),
    )
    print("utility:", {k: round(v, 4) for k, v in evaluation.utility_metrics.items() if "loss" not in k})
    print("privacy summary:", {
        k: round(v, 4)
        for k, v in evaluation.privacy_metrics.items()
        if k in ("clean_attack_auroc", "defended_attack_auroc", "privacy_gain")
    })

    # ---- self-checks -------------------------------------------------------
    raw = output.intermediate_outputs["raw_probabilities"]
    protected = np.asarray(output.protected_outputs)
    assert np.allclose(protected.sum(axis=1), 1.0, atol=1e-6), "rows must stay on the simplex"
    assert (protected >= -1e-9).all(), "probabilities must stay non-negative"
    assert np.array_equal(raw.argmax(axis=1), protected.argmax(axis=1)), "argmax must be preserved"
    assert stats["argmax_preserved_fraction"] == 1.0, "argmax must be preserved for every row"
    assert stats["perturbed_fraction"] >= 0.9, "the adversarial optimization should succeed broadly"
    # The additive term is 2x the solver's crossing tolerance (default 0.01):
    # perturbed rows land within tolerance of the boundary, so the mean gap is
    # dominated by it even when the raw gap itself is small.
    assert (
        stats["surrogate_score_gap_after"]
        <= 0.25 * stats["surrogate_score_gap_before"] + 0.02
    ), "the surrogate attack model should end near its 0.5 decision boundary"

    # ---- signals-only path (precomputed probabilities, no model needed) ----
    with torch.no_grad():
        query_logits = model(data["test_x"][:64])
    # Same batching path the defense itself uses, so the signals-only
    # surrogate is calibrated on identical posteriors.
    member_probs = torch.softmax(
        predict_logits(model, data["train_x"], device="cpu", batch_size=128), dim=1
    )
    nonmember_probs = torch.softmax(
        predict_logits(model, data["reference_x"], device="cpu", batch_size=128), dim=1
    )
    signals_output = MemGuardDefense(device="cpu").run(
        DefenseInput(
            signals={"logits": query_logits},
            auxiliary_data={
                "member_probabilities": member_probs,
                "nonmember_probabilities": nonmember_probs,
            },
        )
    )
    signals_protected = np.asarray(signals_output.protected_outputs)
    assert np.array_equal(
        query_logits.argmax(dim=1).numpy(), signals_protected.argmax(axis=1)
    )
    print("signals-only path OK (64 rows, argmax preserved)")

    # ---- attack benchmark against the toolkit's own attack classes ---------
    query_x = torch.cat([data["train_x"], data["test_x"]])
    query_y = torch.cat([data["train_y"], data["test_y"]])
    membership = torch.cat([torch.ones(300, dtype=torch.long), torch.zeros(400, dtype=torch.long)])

    raw_aurocs = attack_aurocs(model, query_x, query_y, membership)
    protected_aurocs = attack_aurocs(
        output.protected_predictor, query_x, query_y, membership
    )
    for name in raw_aurocs:
        print(
            f"attack {name:>16s}: raw AUROC {raw_aurocs[name]:.3f} -> "
            f"protected {protected_aurocs[name]:.3f} "
            f"(drop {raw_aurocs[name] - protected_aurocs[name]:+.3f})"
        )

    return {
        "train_acc": train_acc,
        "test_acc": test_acc,
        "protected_test_acc": evaluation.utility_metrics["test_accuracy"],
        "clean_attack_auroc": evaluation.privacy_metrics["clean_attack_auroc"],
        "defended_attack_auroc": evaluation.privacy_metrics["defended_attack_auroc"],
        "privacy_gain": evaluation.privacy_metrics["privacy_gain"],
        "perturbed_fraction": stats["perturbed_fraction"],
        "score_gap_after": stats["surrogate_score_gap_after"],
        **{f"raw_{name}": value for name, value in raw_aurocs.items()},
        **{f"protected_{name}": value for name, value in protected_aurocs.items()},
    }


def main() -> None:
    per_seed: List[Dict[str, float]] = [run_seed(seed) for seed in (0, 1, 2)]

    print("\n=== MemGuard benchmark: mean +/- std over seeds 0/1/2 ===")
    header = [
        "train_acc",
        "test_acc",
        "protected_test_acc",
        "perturbed_fraction",
        "clean_attack_auroc",
        "defended_attack_auroc",
        "privacy_gain",
        "raw_confidence",
        "protected_confidence",
        "raw_entropy",
        "protected_entropy",
        "raw_loss",
        "protected_loss",
    ]
    print(f"{'metric':>28s} {'mean':>8s} {'std':>8s}")
    for key in header:
        values = np.asarray([row[key] for row in per_seed])
        print(f"{key:>28s} {values.mean():8.4f} {values.std(ddof=0):8.4f}")

    # Metric-attack separability |AUROC - 0.5| must shrink for the signals the
    # perturbation equalizes (confidence / entropy), and the loss attack must
    # be clearly weakened.
    for signal in ("confidence", "entropy"):
        raw_gap = np.abs(np.asarray([row[f"raw_{signal}"] for row in per_seed]) - 0.5)
        protected_gap = np.abs(np.asarray([row[f"protected_{signal}"] for row in per_seed]) - 0.5)
        assert (protected_gap < raw_gap).all(), f"{signal} attack separability must shrink"
    raw_loss = np.asarray([row["raw_loss"] for row in per_seed])
    protected_loss = np.asarray([row["protected_loss"] for row in per_seed])
    assert (raw_loss - protected_loss > 0.05).all(), "loss attack must be clearly weakened"
    gains = np.asarray([row["privacy_gain"] for row in per_seed])
    assert gains.mean() > 0.05, "MemGuard should clearly reduce the strongest attack AUROC"
    preserved = np.asarray([row["protected_test_acc"] - row["test_acc"] for row in per_seed])
    assert (np.abs(preserved) < 1e-9).all(), "accuracy must be exactly preserved"
    print("\nAll MemGuard self-checks passed.")


if __name__ == "__main__":
    main()
