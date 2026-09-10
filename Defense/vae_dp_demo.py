"""
End-to-end demo for the DP-SGD VAE defence using the project's minimal defense
interface.

Setup
-----
Members and non-members are independent random Gaussian points (every point is
unique). A VAE trained with a small KL weight (β-VAE, near an autoencoder)
memorises individual training points, so members reconstruct with much lower
error than non-members — a reconstruction-based MIA exploits that gap. We train
two VAEs on the same member data:

- a *baseline* VAE with plain Adam (no DP), which overfits and leaks;
- a *defended* VAE trained with DP-SGD (per-sample gradient clipping + Gaussian
  noise via DP-Adam), which fits the distribution without memorising individual
  samples.

The reconstruction-MIA AUROC should drop sharply for the DP VAE (better
privacy), at the cost of higher reconstruction error (the privacy-utility
trade-off).

Note: DP-SGD with exact per-sample gradients (microbatch of 1) is slower than
plain training; the demo takes a couple of minutes on CPU.

Run
---
python Defense/vae_dp_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Defense.base import DefenseInput
from Defense.vae_dp import VAEDPDefense


SEED = 0
DIM = 30
N_TRAIN = 400          # member training set
N_TEST = 400           # non-member set
EPOCHS = 50
KL_WEIGHT = 0.1        # β: small, so the baseline memorises members and leaks
NOISE_MULTIPLIER = 0.5
MAX_GRAD_NORM = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_data():
    rng = np.random.default_rng(SEED)
    members = rng.normal(size=(N_TRAIN, DIM)).astype(np.float32)
    nonmembers = rng.normal(size=(N_TEST, DIM)).astype(np.float32)
    return members, nonmembers


def run_defense(use_dp: bool, members, nonmembers):
    torch.manual_seed(SEED)
    defense = VAEDPDefense(
        latent_dim=16, hidden=(128, 128), use_dp=use_dp,
        epochs=EPOCHS, lr=1e-2, batch_size=64,
        noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM, recon_passes=3,
    )
    defense_input = DefenseInput(
        target_model=None,
        train_data=members,
        test_data=nonmembers,
        defense_config={
            "use_dp": use_dp, "epochs": EPOCHS, "batch_size": 64, "lr": 1e-2,
            "latent_dim": 16, "hidden": (128, 128), "kl_weight": KL_WEIGHT,
            "noise_multiplier": NOISE_MULTIPLIER, "max_grad_norm": MAX_GRAD_NORM,
            "recon_passes": 3,
        },
        eval_config={"enabled": True},
        metadata={"variant": "dp" if use_dp else "baseline"},
    )
    return defense.run(defense_input)


def report(name, output):
    ev = output.evaluation
    util = ev.utility_metrics or {}
    priv = ev.privacy_metrics or {}
    print(f"\n[{name}]")
    print(f"  reconstruction MSE  train={util.get('reconstruction_mse_train', float('nan')):.4f}  "
          f"test={util.get('reconstruction_mse_test', float('nan')):.4f}")
    print(f"  reconstruction-MIA  AUROC={priv.get('mia_auroc', float('nan')):.3f}  "
          f"TPR@5%FPR={priv.get('mia_auroc_low_fpr', float('nan')):.3f}")
    return ev


def main() -> None:
    members, nonmembers = build_data()
    print("=" * 64)
    print("DP-SGD VAE defence demo (reconstruction-MIA privacy-utility trade-off)")
    print("=" * 64)
    print(f"members={N_TRAIN}  non-members={N_TEST}  dim={DIM}  epochs={EPOCHS}  beta={KL_WEIGHT}")

    out_base = run_defense(use_dp=False, members=members, nonmembers=nonmembers)
    ev_base = report("baseline (plain Adam, no DP)", out_base)

    out_dp = run_defense(use_dp=True, members=members, nonmembers=nonmembers)
    ev_dp = report("defended (DP-Adam)", out_dp)

    print("=" * 64)
    base_auroc = (ev_base.privacy_metrics or {}).get("mia_auroc")
    dp_auroc = (ev_dp.privacy_metrics or {}).get("mia_auroc")
    base_test = (ev_base.utility_metrics or {}).get("reconstruction_mse_test")
    dp_test = (ev_dp.utility_metrics or {}).get("reconstruction_mse_test")
    print(f"\nprivacy: MIA AUROC {base_auroc:.3f} -> {dp_auroc:.3f}  (lower is better)")
    print(f"utility: test MSE  {base_test:.3f} -> {dp_test:.3f}   (lower is better)")

    # Self-checks: baseline leaks, DP reduces leakage.
    assert base_auroc > 0.55, (
        f"baseline VAE should leak (AUROC > 0.55), got {base_auroc:.3f}"
    )
    assert dp_auroc < base_auroc, (
        f"DP-SGD should reduce MIA AUROC ({base_auroc:.3f} -> got {dp_auroc:.3f})"
    )
    print("All self-checks passed (DP-SGD reduced membership leakage).")


if __name__ == "__main__":
    main()
