"""
End-to-end demo for the GAN-Leaks attack using the project's minimal attack
interface.

We do not train a full GAN (slow and finicky on CPU). Instead we train a small
autoencoder on the member data and treat its decoder as the generator ``G(z)``.
This faithfully captures the property GAN-Leaks exploits — the generator
reconstructs the distribution it was trained on (members) far better than
out-of-distribution non-members — and lets us exercise both attack modes:

- FBB (Full Black-Box): k-NN distance between queries and generated samples.
- PBB (Partial Black-Box): per-query latent inversion through the generator.

Members are drawn from one Gaussian region, non-members from a shifted
(out-of-distribution) region, so reconstruction cleanly separates the two.

Run
---
python Attack/gan_leaks_demo.py
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

from Attack.base import AttackInput
from Attack.gan_leaks import GANLeaksAttack


SEED = 0
DIM = 32
LATENT = 8
N_TRAIN_AE = 1500          # members used to train the autoencoder
N_GEN = 800                # generated samples for FBB
N_QUERY_PER_SIDE = 250     # members + non-members to attack
AE_EPOCHS = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AutoEncoder(nn.Module):
    def __init__(self, dim: int, latent: int) -> None:
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


def build_data():
    rng = np.random.default_rng(SEED)
    # Members: tight Gaussian at the origin. Non-members: shifted (OOD) Gaussian.
    members = rng.normal(0.0, 0.5, size=(N_TRAIN_AE + 2 * N_QUERY_PER_SIDE, DIM)).astype(np.float32)
    nonmembers = rng.normal(5.0, 0.5, size=(2 * N_QUERY_PER_SIDE, DIM)).astype(np.float32)

    ae_train = members[:N_TRAIN_AE]
    remaining = members[N_TRAIN_AE:]
    gen_pool, query_mem = remaining[:N_QUERY_PER_SIDE], remaining[N_QUERY_PER_SIDE:2 * N_QUERY_PER_SIDE]
    calib_nonmem, query_nonmem = nonmembers[:N_QUERY_PER_SIDE], nonmembers[N_QUERY_PER_SIDE:]
    return rng, ae_train, gen_pool, query_mem, query_nonmem, calib_nonmem


def train_autoencoder(X_train) -> AutoEncoder:
    torch.manual_seed(SEED)
    ae = AutoEncoder(DIM, LATENT).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    X = torch.from_numpy(X_train).to(DEVICE)
    for _ in range(AE_EPOCHS):
        opt.zero_grad()
        loss_fn(ae(X), X).backward()
        opt.step()
    return ae


def generated_samples(ae: AutoEncoder, gen_pool, n_gen: int) -> np.ndarray:
    """Decode latents drawn from the encoded member posterior -> generated samples."""
    ae.eval()
    with torch.no_grad():
        z = ae.enc(torch.from_numpy(gen_pool).to(DEVICE)).cpu().numpy()
    mu, std = z.mean(0), z.std(0) + 1e-3
    rng = np.random.default_rng(SEED + 1)
    z_sample = rng.normal(mu, std, size=(n_gen, LATENT)).astype(np.float32)
    with torch.no_grad():
        gen = ae.dec(torch.from_numpy(z_sample).to(DEVICE)).cpu().numpy()
    return gen.astype(np.float32)


def run_mode(mode, target_model, generated, query_mem, query_nonmem, calib_nonmem, **cfg):
    X_query = np.concatenate([query_mem, query_nonmem])
    labels = np.concatenate([np.ones(len(query_mem)), np.zeros(len(query_nonmem))]).astype(np.int64)
    attack_input = AttackInput(
        target_model=target_model,
        samples=X_query,
        signals={"generated_samples": generated} if mode == "fbb" else None,
        reference_data={"calibration_X": calib_nonmem},
        membership_labels=labels,
        config={"mode": mode, **cfg},
    )
    return GANLeaksAttack(mode=mode).run(attack_input), labels


def report(name, output, labels):
    errors = output.intermediate_outputs["reconstruction_error"]
    members = labels == 1
    preds = np.asarray(output.membership_preds)
    print(f"\n[{name}] mode={output.intermediate_outputs['mode']}")
    print(f"  reconstruction error  members={errors[members].mean():.3f}  "
          f"non-members={errors[~members].mean():.3f}")
    print(f"  AUROC={output.evaluation.auroc:.3f}  pred_accuracy={(preds == labels).mean():.3f}")
    print(f"  TPR@1%FPR={output.evaluation.tpr_at_fpr['1%']:.3f}  "
          f"TPR@0.1%FPR={output.evaluation.tpr_at_fpr['0.1%']:.3f}")
    return output


def main() -> None:
    torch.manual_seed(SEED)
    rng, ae_train, gen_pool, query_mem, query_nonmem, calib_nonmem = build_data()
    ae = train_autoencoder(ae_train)
    generated = generated_samples(ae, gen_pool, N_GEN)

    print("=" * 64)
    print("GAN-Leaks demo (autoencoder decoder as generator)")
    print("=" * 64)

    out_fbb, labels = run_mode(
        "fbb", target_model=None, generated=generated,
        query_mem=query_mem, query_nonmem=query_nonmem, calib_nonmem=calib_nonmem, k=5,
    )
    report("FBB", out_fbb, labels)

    out_pbb, _ = run_mode(
        "pbb", target_model=ae.dec, generated=generated,
        query_mem=query_mem, query_nonmem=query_nonmem, calib_nonmem=calib_nonmem,
        z_dim=LATENT, n_steps=300, lr=0.1,
    )
    report("PBB", out_pbb, labels)

    print("=" * 64)
    # Self-checks (design doc §7.1, §10.3: higher score -> more likely member).
    for name, out in [("FBB", out_fbb), ("PBB", out_pbb)]:
        errors = out.intermediate_outputs["reconstruction_error"]
        members = labels == 1
        assert errors[members].mean() < errors[~members].mean(), (
            f"{name}: members should have lower reconstruction error"
        )
        assert out.evaluation is not None and out.evaluation.auroc is not None
        assert out.evaluation.auroc > 0.8, (
            f"{name}: AUROC expected > 0.8, got {out.evaluation.auroc:.3f}"
        )
    print("All self-checks passed (both FBB and PBB).")


if __name__ == "__main__":
    main()
