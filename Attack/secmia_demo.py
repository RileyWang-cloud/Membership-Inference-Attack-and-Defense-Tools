"""
Real-data demo for the unified SecMIA interface.

This demo directly embeds the test parameters in the file, similar to other
project demos. It drives the real `Attack/utils_secmia/secmia.py` workflow
through the unified `SecMIAAttack` wrapper.

Before running, update the constants in the "Demo Config" section if your
checkpoint or dataset location differs.

Run
---
python Attack/secmia_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.secmia import AttackInput, SecMIAAttack
from Attack.utils_secmia.mia_evals.dataset_utils import load_member_data


# Demo Config
UTILS_SECMIA_DIR = Path(__file__).resolve().parent / "utils_secmia"
DATA_ROOT = str(UTILS_SECMIA_DIR / "datasets")
MODEL_DIR = str(UTILS_SECMIA_DIR / "logs" / "DDPM_CIFAR10")
DATASET = "CIFAR10"
CKPT_NAME = "ckpt-step800000.pt"
MODEL_TYPE = "ddpm"
ATTACK_VARIANT = "stat"
T_SEC = 100
M = 10
BATCH_SIZE = 128
OUTPUT_SAVE_DIR = str(UTILS_SECMIA_DIR / "experiment_results" / "DDPM_CIFAR10")
K = 3
NUM_T_GROUPS = 5
SPARSITY = 0.3
NNS_TRAIN_PORTION = 0.2
# FID measures target-model generation quality, independently of MIA quality.
# 100 images is a quick smoke test; use at least 10,000 for a final report.
COMPUTE_FID = True
FID_NUM_IMAGES = 2000
FID_BATCH_SIZE = 32
FID_CACHE = str(UTILS_SECMIA_DIR / "stats" / "cifar10.train.npz")
FID_USE_TORCH = False


def build_membership_labels(data_root: str, dataset: str, batch_size: int) -> np.ndarray:
    member_set, nonmember_set, _, _ = load_member_data(
        dataset_root=data_root,
        dataset_name=dataset,
        batch_size=batch_size,
        shuffle=False,
        randaugment=False,
    )
    return np.concatenate(
        [
            np.ones(len(member_set), dtype=np.int64),
            np.zeros(len(nonmember_set), dtype=np.int64),
        ],
        axis=0,
    )


def main() -> None:
    membership_labels = build_membership_labels(
        data_root=DATA_ROOT,
        dataset=DATASET,
        batch_size=BATCH_SIZE,
    )

    attack_input = AttackInput(
        target_model=None,
        samples=None,
        membership_labels=membership_labels,
        config={
            "attack_variant": ATTACK_VARIANT,
            "model_dir": MODEL_DIR,
            "data_root": DATA_ROOT,
            "dataset": DATASET,
            "t_sec": T_SEC,
            "m": M,
            "model_type": MODEL_TYPE,
            "ckpt_name": CKPT_NAME,
            "output_save_dir": OUTPUT_SAVE_DIR,
            "batch_size": BATCH_SIZE,
            "k": K,
            "num_t_groups": NUM_T_GROUPS,
            "sparsity": SPARSITY,
            "nns_train_portion": NNS_TRAIN_PORTION,
            "compute_fid": COMPUTE_FID,
            "fid_num_images": FID_NUM_IMAGES,
            "fid_batch_size": FID_BATCH_SIZE,
            "fid_cache": FID_CACHE,
            "fid_use_torch": FID_USE_TORCH,
        },
        metadata={
            "model_dir": MODEL_DIR,
            "ckpt_name": CKPT_NAME,
            "flag_path": str(Path(MODEL_DIR) / "flagfile.txt"),
        },
    )

    attack = SecMIAAttack(
        attack_variant=ATTACK_VARIANT,
        t_sec=T_SEC,
        m=M,
        batch_size=BATCH_SIZE,
        model_type=MODEL_TYPE,
        k=K,
        num_t_groups=NUM_T_GROUPS,
        sparsity=SPARSITY,
        output_save_dir=OUTPUT_SAVE_DIR,
        nns_train_portion=NNS_TRAIN_PORTION,
        compute_fid=COMPUTE_FID,
        fid_num_images=FID_NUM_IMAGES,
        fid_batch_size=FID_BATCH_SIZE,
        fid_cache=FID_CACHE,
        fid_use_torch=FID_USE_TORCH,
    )
    output = attack.run(attack_input)

    print("SecMIA real demo finished.")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Checkpoint: {CKPT_NAME}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Dataset: {DATASET}")
    print(f"Attack variant: {ATTACK_VARIANT}")
    if "fid" in output.metadata:
        print(f"FID ({FID_NUM_IMAGES} generated images): {output.metadata['fid']:.4f}")
    print(f"Total attacked samples: {len(membership_labels)}")
    print(f"Scores shape: {np.asarray(output.membership_scores).shape}")
    if output.evaluation is not None:
        print(f"AUROC: {output.evaluation.auroc:.4f}" if output.evaluation.auroc is not None else "AUROC: None")
        print(f"Accuracy: {output.evaluation.accuracy:.4f}")
        print(f"TPR@1%FPR: {output.evaluation.tpr_at_fpr['1%']:.4f}")
        print(f"TPR@0.1%FPR: {output.evaluation.tpr_at_fpr['0.1%']:.4f}")
        if output.evaluation.extra_metrics is not None and "selected_threshold" in output.evaluation.extra_metrics:
            print(f"Selected threshold: {output.evaluation.extra_metrics['selected_threshold']:.6f}")
    print("First 5 membership scores:", np.asarray(output.membership_scores)[:5])
    if output.intermediate_outputs is not None:
        print("Intermediate outputs:", sorted(output.intermediate_outputs.keys()))


if __name__ == "__main__":
    main()
