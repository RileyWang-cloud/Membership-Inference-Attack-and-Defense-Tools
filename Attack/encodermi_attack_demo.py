"""
End-to-end demo for vector-based EncoderMI using the project's minimal
attack interface.

1. Build synthetic member images (smooth patterns) and non-member images
   (white noise).
2. Train a shadow encoder and a target encoder on disjoint member splits
   (reconstruction objective against a fixed random decoder).
3. Fit the binary attack model on the shadow encoder's sorted
   augmentation-consistency features.
4. Attack a member/non-member mixture with the target encoder and evaluate.

Run
---
python Attack/encodermi_attack_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.base import AttackInput
from Attack.encodermi_attack import EncoderMIAttack

torch.manual_seed(0)
np.random.seed(0)

IMG_SHAPE = (1, 16, 16)
FEAT_DIM = 64


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(np.prod(IMG_SHAPE)), 128),
            nn.ReLU(),
            nn.Linear(128, FEAT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


def make_member_images(n: int) -> torch.Tensor:
    """Structured 16x16 images: upscaled smooth random blocks."""
    base = torch.randn(n, 1, 4, 4)
    imgs = F.interpolate(base, size=(16, 16), mode="bilinear")
    return imgs.clamp(-1, 1)


def make_nonmember_images(n: int) -> torch.Tensor:
    return torch.randn(n, *IMG_SHAPE).clamp(-1, 1)


def train_encoder(member_images, epochs=200, lr=1e-2) -> TinyEncoder:
    """Reconstruction against a fixed random decoder: the encoder learns the
    member subspace, so in-distribution samples map consistently."""
    encoder = TinyEncoder()
    decoder_w = torch.randn(FEAT_DIM, int(np.prod(IMG_SHAPE))) * 0.1
    optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    targets = member_images.reshape(len(member_images), -1)

    encoder.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = encoder(member_images) @ decoder_w
        loss = F.mse_loss(recon, targets)
        loss.backward()
        optimizer.step()
    encoder.eval()
    return encoder


def main() -> None:
    # 1. data: disjoint shadow / target member pools
    shadow_members = make_member_images(200)
    shadow_nonmembers = make_nonmember_images(200)
    target_members = make_member_images(200)
    target_nonmembers = make_nonmember_images(200)

    shadow_encoder = train_encoder(shadow_members)
    target_encoder = train_encoder(target_members)

    # 2. attack mixture under the target encoder
    samples = torch.cat([target_members, target_nonmembers])
    membership_labels = np.concatenate(
        [np.ones(200, dtype=np.int64), np.zeros(200, dtype=np.int64)]
    )

    attack = EncoderMIAttack(n_augmentations=10, epochs=60)
    attack_input = AttackInput(
        target_model=target_encoder,
        samples=samples,
        membership_labels=membership_labels,
        shadow_data={
            "shadow_model": shadow_encoder,
            "member_samples": shadow_members,
            "nonmember_samples": shadow_nonmembers,
        },
        config={"n_augmentations": 10, "batch_size": 128},
    )

    output = attack.run(attack_input)

    # 3. report
    scores = np.asarray(output.membership_scores)
    features = output.intermediate_outputs["consistency_features"]
    mean_top = features.mean(axis=1) # mean sorted pairwise cosine similarity
    print("EncoderMI (vector-based) demo")
    print(f"  feature dim        : {output.metadata['feature_dim']}")
    print(f"  mean sim (member)  : {mean_top[membership_labels == 1].mean():.4f}")
    print(f"  mean sim (non-mem) : {mean_top[membership_labels == 0].mean():.4f}")
    print(f"  score mean (member): {scores[membership_labels == 1].mean():.4f}")
    print(f"  score mean (non-mb): {scores[membership_labels == 0].mean():.4f}")
    print(f"  auroc (sklearn)    : {roc_auc_score(membership_labels, scores):.4f}")
    eval_result = output.evaluation
    if eval_result is not None:
        print(f"  accuracy           : {eval_result.accuracy:.4f}")
        print(f"  auroc (unified)    : {eval_result.auroc:.4f}")
        print(f"  tpr@1% fpr         : {eval_result.tpr_at_fpr['1%']:.4f}")


if __name__ == "__main__":
    main()
