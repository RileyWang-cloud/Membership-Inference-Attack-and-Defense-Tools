"""
Real-data demo for the unified GSAMIA interface.

This demo embeds the test parameters directly in the file and drives the real
`Attack/utils_gsamia/gsamia.py` gradient-feature extraction workflow through
the unified `GSAMIAAttack` wrapper.

The demo uses:
- one target checkpoint for the attacked model
- one shadow checkpoint for training the attack model
- the benchmark member/non-member split loaded from `utils_gsamia/datasets`

Run
---
python Attack/gsamia_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Attack.gsamia import AttackInput, GSAMIAAttack
from Attack.utils_gsamia import gsamia as gsamia_utils


# Demo Config
UTILS_GSAMIA_DIR = Path(__file__).resolve().parent / "utils_gsamia"
DATA_ROOT = str(UTILS_GSAMIA_DIR / "datasets")
MODEL_DIR = str(UTILS_GSAMIA_DIR / "logs" / "DDPM_CIFAR10")
DATASET = "cifar10"
MODEL_TYPE = "ddpm"
TARGET_CKPT_NAME = "ckpt-step800000.pt"
SHADOW_CKPT_NAME = "ckpt-step600000.pt"
ATTACK_METHOD = 1
SAMPLING_FREQUENCY = 5
PREDICTION_TYPE = "epsilon"
BATCH_SIZE = 16
K = 3
NUM_T_GROUPS = 5
SPARSITY = 0.3
XGB_N_ESTIMATORS = 200
USE_CACHED_FEATURES = True
# FID measures target-model generation quality, independently of MIA quality.
# 100 images is a quick smoke test; use at least 10,000 for a final report.
COMPUTE_FID = True
FID_NUM_IMAGES = 2000
FID_BATCH_SIZE = 16
FID_CACHE = str(UTILS_GSAMIA_DIR / "stats" / "cifar10.train.npz")
FID_USE_TORCH = False

TARGET_OUTPUT_NAME = str(UTILS_GSAMIA_DIR / "gradient_mia" / "GSA1" / "target_feature_DDPM")
SHADOW_OUTPUT_NAME = str(UTILS_GSAMIA_DIR / "gradient_mia" / "GSA1" / "shadow_feature_DDPM")


def build_membership_labels(member_loader: object, nonmember_loader: object) -> np.ndarray:
    """
    Build labels aligned with the current GSAMIA feature extractor.

    The underlying `utils_gsamia.gsamia.extract_attack_features()` appends one
    gradient feature vector per dataloader batch, not per raw sample. So the
    final attack-score length matches `len(loader)` rather than
    `len(loader.dataset)`.
    """
    member_size = len(member_loader) # type: ignore[arg-type]
    nonmember_size = len(nonmember_loader) # type: ignore[arg-type]
    return np.concatenate(
        [
            np.ones(member_size, dtype=np.int64),
            np.zeros(nonmember_size, dtype=np.int64),
        ],
        axis=0,
    )


def load_model(ckpt_name: str):
    flag_path = Path(MODEL_DIR) / "flagfile.txt"
    flags_obj = gsamia_utils.get_FLAGS(str(flag_path))
    flags_obj(sys.argv[:1] or ["gsamia_demo.py"])

    ckpt_path = str(Path(MODEL_DIR) / ckpt_name)
    if MODEL_TYPE == "ddpm":
        model = gsamia_utils.get_model(ckpt_path, flags_obj, WA=True)
    elif MODEL_TYPE == "etdm":
        model = gsamia_utils.get_model_grouped(ckpt_path, flags_obj)
    elif MODEL_TYPE == "smcd":
        model = gsamia_utils.get_model_selective(ckpt_path, flags_obj)
    else:
        raise ValueError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")
    return model, flags_obj


def main() -> None:
    shadow_model, _ = load_model(SHADOW_CKPT_NAME)
    member_loader, nonmember_loader = gsamia_utils.get_dataset(
        dataset_root=DATA_ROOT,
        dataset=DATASET,
        batch_size=BATCH_SIZE,
    )
    membership_labels = build_membership_labels(member_loader, nonmember_loader)

    attack_input = AttackInput(
        target_model=None,
        samples=None,
        membership_labels=membership_labels,
        shadow_data={
            "shadow_model": shadow_model,
            "member_samples": member_loader,
            "nonmember_samples": nonmember_loader,
            "output_name": SHADOW_OUTPUT_NAME,
            "ckpt_name": SHADOW_CKPT_NAME,
        },
        config={
            "model_dir": MODEL_DIR,
            "data_root": DATA_ROOT,
            "dataset": DATASET,
            "model_type": MODEL_TYPE,
            "ckpt_name": TARGET_CKPT_NAME,
            "attack_method": ATTACK_METHOD,
            "sampling_frequency": SAMPLING_FREQUENCY,
            "prediction_type": PREDICTION_TYPE,
            "batch_size": BATCH_SIZE,
            "k": K,
            "num_t_groups": NUM_T_GROUPS,
            "sparsity": SPARSITY,
            "xgb_n_estimators": XGB_N_ESTIMATORS,
            "use_cached_features": USE_CACHED_FEATURES,
            "output_name": TARGET_OUTPUT_NAME,
            "compute_fid": COMPUTE_FID,
            "fid_num_images": FID_NUM_IMAGES,
            "fid_batch_size": FID_BATCH_SIZE,
            "fid_cache": FID_CACHE,
            "fid_use_torch": FID_USE_TORCH,
        },
        metadata={
            "model_dir": MODEL_DIR,
            "ckpt_name": TARGET_CKPT_NAME,
            "flag_path": str(Path(MODEL_DIR) / "flagfile.txt"),
            "output_name": TARGET_OUTPUT_NAME,
        },
    )

    attack = GSAMIAAttack(
        attack_method=ATTACK_METHOD,
        sampling_frequency=SAMPLING_FREQUENCY,
        prediction_type=PREDICTION_TYPE,
        batch_size=BATCH_SIZE,
        model_type=MODEL_TYPE,
        output_name=TARGET_OUTPUT_NAME,
        xgb_n_estimators=XGB_N_ESTIMATORS,
        use_cached_features=USE_CACHED_FEATURES,
        compute_fid=COMPUTE_FID,
        fid_num_images=FID_NUM_IMAGES,
        fid_batch_size=FID_BATCH_SIZE,
        fid_cache=FID_CACHE,
        fid_use_torch=FID_USE_TORCH,
    )
    output = attack.run(attack_input)

    print("GSAMIA real demo finished.")
    print(f"Model dir: {MODEL_DIR}")
    print(f"Target checkpoint: {TARGET_CKPT_NAME}")
    print(f"Shadow checkpoint: {SHADOW_CKPT_NAME}")
    print(f"Data root: {DATA_ROOT}")
    print(f"Dataset: {DATASET}")
    print(f"Attack method: {ATTACK_METHOD}")
    print(f"Sampling frequency: {SAMPLING_FREQUENCY}")
    print(f"Use cached features: {USE_CACHED_FEATURES}")
    if "fid" in output.metadata:
        print(f"FID ({FID_NUM_IMAGES} generated images): {output.metadata['fid']:.4f}")
    print(f"Total attacked samples: {len(membership_labels)}")
    print(f"Scores shape: {np.asarray(output.membership_scores).shape}")
    if output.evaluation is not None:
        print(f"AUROC: {output.evaluation.auroc:.4f}" if output.evaluation.auroc is not None else "AUROC: None")
        print(f"Accuracy: {output.evaluation.accuracy:.4f}")
        print(f"TPR@1%FPR: {output.evaluation.tpr_at_fpr['1%']:.4f}")
        print(f"TPR@0.1%FPR: {output.evaluation.tpr_at_fpr['0.1%']:.4f}")
    print("First 5 membership scores:", np.asarray(output.membership_scores)[:5])
    if output.intermediate_outputs is not None:
        print("Intermediate outputs:", sorted(output.intermediate_outputs.keys()))


if __name__ == "__main__":
    main()
