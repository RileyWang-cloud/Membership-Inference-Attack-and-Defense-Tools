"""Train ADF/SMCD and evaluate FID, SecMIA, and optionally GSAMIA."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_SECMIA_DIR = REPO_ROOT / "Attack" / "utils_secmia"
UTILS_GSAMIA_DIR = REPO_ROOT / "Attack" / "utils_gsamia"
# The imported research utilities use legacy absolute imports such as
# ``from score import fid`` and ``from mia_evals...``.  Keep their directories
# on sys.path; the folders themselves remain under Attack/.
for module_dir in (REPO_ROOT, UTILS_SECMIA_DIR, UTILS_GSAMIA_DIR):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

from Defense.adf import ADFDefense
from Defense.base import DefenseInput
from Attack.utils_secmia.mia_evals.dataset_utils import load_member_data
from Attack.utils_secmia.model import UNet
from Attack.utils_gsamia import gsamia as gsamia_utils

DATA_ROOT = REPO_ROOT / "Attack" / "utils_secmia" / "datasets"
SAVE_DIR = REPO_ROOT / "Defense" / "artifacts" / "adf_cifar10"
TARGET_CKPT_PATH = SAVE_DIR / "adf_smcd.pt"
# Set False only when you explicitly want to train ADF from scratch.
SKIP_DEFENSE_TRAINING = True
DATASET = "CIFAR10"
DIFFUSION_STEPS = 1000
TRAINING_STEPS = 1000
K = 10
NUM_T_GROUPS = 1
SPARSITY = 0.5
# This stage is O(diffusion_steps * K); use 1 only as a minimal reference run.
STAGE1_STEPS = 1
COMPUTE_FID = True
FID_NUM_IMAGES = 2000
FID_BATCH_SIZE = 4
RUN_ATTACKS = True
SECMIA_BATCH_SIZE = 16
GSAMIA_BATCH_SIZE = 1
GSAMIA_SAMPLING_FREQUENCY = 5
GSAMIA_TIMESTEP_CHUNK_SIZE = 1
# Black-box threat model: the attacker does not know ADF and trains a plain DDPM
# shadow model, exactly as in Attack/gsamia_demo.py.
SHADOW_MODEL_DIR = REPO_ROOT / "Attack" / "utils_gsamia" / "logs" / "DDPM_CIFAR10"
SHADOW_CKPT_NAME = "ckpt-step600000.pt"


def model_factory():
    return UNet(T=DIFFUSION_STEPS, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], num_res_blocks=2, dropout=0.1)


def load_shadow_model():
    """Load the existing DDPM shadow checkpoint used by gsamia_demo.py."""
    ckpt_path = SHADOW_MODEL_DIR / SHADOW_CKPT_NAME
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"GSAMIA DDPM shadow checkpoint was not found: {ckpt_path}"
        )
    flags_obj = SimpleNamespace(
        T=DIFFUSION_STEPS, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1, num_t_groups=NUM_T_GROUPS,
    )
    return gsamia_utils.get_model(str(ckpt_path), flags_obj, WA=True)


def main() -> None:
    member_set, _, _, _ = load_member_data(str(DATA_ROOT), DATASET, batch_size=128, shuffle=False)
    shadow_model = load_shadow_model() if RUN_ATTACKS else None
    defense = ADFDefense(
        diffusion_steps=DIFFUSION_STEPS, training_steps=TRAINING_STEPS, k=K,
        num_t_groups=NUM_T_GROUPS, sparsity=SPARSITY, stage1_steps=STAGE1_STEPS,
        save_dir=str(SAVE_DIR), checkpoint_name="adf_smcd.pt",
    )
    output = defense.run(DefenseInput(
        model_factory=model_factory,
        train_data=member_set,
        defense_config={
            "skip_training": SKIP_DEFENSE_TRAINING,
            "checkpoint_path": str(TARGET_CKPT_PATH),
        },
        eval_config={
            "compute_fid": COMPUTE_FID, "fid_num_images": FID_NUM_IMAGES,
            "fid_batch_size": FID_BATCH_SIZE,
            "fid_cache": str(REPO_ROOT / "Attack" / "utils_secmia" / "stats" / "cifar10.train.npz"),
            "run_attacks": RUN_ATTACKS, "data_root": str(DATA_ROOT), "dataset": DATASET,
            "attack_batch_size": SECMIA_BATCH_SIZE,
            "model_type": "smcd", "gsamia_shadow_model": shadow_model,
            "gsamia_batch_size": GSAMIA_BATCH_SIZE,
            "gsamia_sampling_frequency": GSAMIA_SAMPLING_FREQUENCY,
            "gsamia_timestep_chunk_size": GSAMIA_TIMESTEP_CHUNK_SIZE,
        },
    ))
    print("ADF defense finished.")
    print("Defended checkpoint:", output.metadata["model_save_path"])
    if output.evaluation is not None:
        print("FID:", (output.evaluation.utility_metrics or {}).get("fid"))
        for name, result in (output.evaluation.extra_metrics or {}).get("attack_results", {}).items():
            print(f"{name}:", result)


if __name__ == "__main__":
    main()
