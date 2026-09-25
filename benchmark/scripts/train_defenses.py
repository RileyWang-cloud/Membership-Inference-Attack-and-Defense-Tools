"""STEP 8-9 (plan §15, W9): train/apply D01-D08 and re-cache partition outputs.

Hyperparameter alignment (plan §10.2): every training-time defense inherits
the clean-target recipe (§5.3) via explicit defense_config overrides — no
reliance on in-code defaults. Deviations are limited to what the algorithm
requires and are recorded in each defense row:

- D01 DP-SGD: GroupNorm ResNet18 variant (plan §10.4) and the repo's
  hard-coded Adam optimizer (architecture + optimizer deviation recorded).
- D05 AdvReg: repo impl has no LR scheduler (deviation recorded).
- D06 MemGuard: inference-time, wraps the CLEAN target checkpoint with the
  posterior-perturbation predictor; surrogate trained on Shadow Bundle
  posteriors only (red line §10.3 / §11.8).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from benchmark.bmk import common
from benchmark.bmk.data import gather_global, load_cifar10_tensors
from benchmark.bmk.models import resnet18_cifar_factory, resnet18_cifar_groupnorm_factory
from benchmark.bmk.train import RECIPE, cache_partition_outputs, partition_indices

# clean-recipe overrides shared by every training-time defense (§10.2)
CLEAN_RECIPE_CONFIG = {
    "batch_size": int(RECIPE["batch_size"]),
    "epochs": int(RECIPE["epochs"]),
    "learning_rate": float(RECIPE["lr"]),
    "momentum": float(RECIPE["momentum"]),
    "weight_decay": float(RECIPE["weight_decay"]),
    "milestones": tuple(RECIPE["scheduler"]["milestones"]),
    "gamma": float(RECIPE["scheduler"]["gamma"]),
    "optimizer": "sgd",
}

DEFENSE_ORDER = [
    "D01_DP_SGD", "D02_RelaxLoss", "D03_HAMP", "D04_EarlyStop",
    "D05_AdvReg", "D06_MemGuard", "D07_Mixup", "D08_LabelSmoothing",
]

DEFENSE_DEVIATIONS = {
    "D01_DP_SGD": "GroupNorm(8) ResNet18 variant (plan §10.4); repo DP-SGD hard-codes Adam lr=1e-3 (no SGD option); per-sample autograd clipping, no privacy accounting (report C=1.0, sigma=1.0, delta=n/a)",
    "D02_RelaxLoss": "none (relax-loss alternation is algorithm-required)",
    "D03_HAMP": "none (entropy penalty / output modification are algorithm-required)",
    "D04_EarlyStop": "patience-based early stopping on target_val (val_loss, patience=10) instead of full 100 epochs",
    "D05_AdvReg": "repo AdvReg has no LR scheduler (clean recipe uses MultiStepLR[50,75]); adversary sub-config at repo defaults",
    "D06_MemGuard": "inference-time defense: wraps the clean target checkpoint; surrogate + noise use Shadow Bundle posteriors only",
    "D07_Mixup": "none (mixup augmentation is algorithm-required)",
    "D08_LabelSmoothing": "none (smoothing alpha=0.1 is algorithm-required)",
}


def _training_input(defense_input_cls, manifest, tensors, model_factory):
    parts = manifest["partitions"]
    X_train, y_train = gather_global(parts["target_train"]["global_indices"], tensors)
    X_val, y_val = gather_global(parts["target_val"]["global_indices"], tensors)
    return defense_input_cls(
        model_factory=model_factory,
        train_data=X_train, train_labels=y_train,
        val_data=X_val, val_labels=y_val,
    )


def run(seed: int, defenses: list[str], epochs_override: int | None) -> None:
    device = common.resolve_device()
    manifest = common.load_manifest(seed)
    tensors = load_cifar10_tensors()
    parts = manifest["partitions"]

    for def_id in defenses:
        tag = f"{def_id}_seed{seed}"
        if common.is_done("defense", tag):
            print(f"[skip] {tag}")
            continue
        t0 = time.time()
        config = dict(CLEAN_RECIPE_CONFIG)
        if epochs_override is not None:
            config["epochs"] = epochs_override

        if def_id == "D01_DP_SGD":
            from Defense.dp_sgd import DPSGDDefense
            from Defense.base import DefenseInput

            dp_config = {
                "batch_size": config["batch_size"],
                "epochs": config["epochs"],
                "learning_rate": 1e-3,        # repo DP-SGD is Adam-fixed (deviation)
                "noise_multiplier": 1.0,
                "max_grad_norm": 1.0,
            }
            defense = DPSGDDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_groupnorm_factory)
            defense_input.defense_config = dp_config
            defense.fit(defense_input)
            model = defense.defended_model

        elif def_id == "D02_RelaxLoss":
            from Defense.base import DefenseInput
            from Defense.relax_loss import RelaxLossDefense

            defense = RelaxLossDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.defense_config = config
            defense.fit(defense_input)
            model = defense.defended_model

        elif def_id == "D03_HAMP":
            from Defense.base import DefenseInput
            from Defense.hamp import HAMPDefense

            defense = HAMPDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.defense_config = config
            common.set_global_seed(seed)  # HAMP draws random reference inputs
            defense.fit(defense_input)
            model = defense.protected_predictor if defense.protected_predictor is not None \
                else defense.defended_model

        elif def_id == "D04_EarlyStop":
            from Defense.base import DefenseInput
            from Defense.early_stop import EarlyStopDefense

            es_config = dict(config)
            es_config.update({"monitor": "val_loss", "patience": 10, "restore_best": True})
            defense = EarlyStopDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.defense_config = es_config
            common.set_global_seed(seed)
            defense.fit(defense_input)
            model = defense.defended_model

        elif def_id == "D05_AdvReg":
            from Defense.base import DefenseInput
            from Defense.adv_reg import AdvRegDefense

            X_aux, y_aux = gather_global(parts["auxiliary"]["global_indices"], tensors)
            defense = AdvRegDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.auxiliary_data = {
                "nonmember_data": X_aux, "nonmember_labels": y_aux,
            }
            defense_input.defense_config = config
            common.set_global_seed(seed)
            defense.fit(defense_input)
            model = defense.defended_model

        elif def_id == "D06_MemGuard":
            from Defense.base import DefenseInput
            from Defense.memguard import MemGuardDefense

            clean_ckpt = torch.load(common.checkpoint_dir("targets") / f"target_seed{seed}.pt",
                                    map_location="cpu", weights_only=False)
            target = resnet18_cifar_factory()
            target.load_state_dict(clean_ckpt["state_dict"])
            shadow = np.load(common.bundle_dir(seed) / "shadow_bundle.npz")
            defense = MemGuardDefense()
            defense_input = DefenseInput(
                target_model=target,
                auxiliary_data={
                    "member_probabilities": shadow["shadow_member_outputs"],
                    "nonmember_probabilities": shadow["shadow_nonmember_outputs"],
                },
            )
            common.set_global_seed(seed)
            defense.fit(defense_input)
            # one infer call constructs the argmax-preserving protected predictor
            X_val, y_val = gather_global(parts["target_val"]["global_indices"], tensors)
            output = defense.infer(DefenseInput(target_model=target, samples=X_val, labels=y_val))
            model = output.protected_predictor

        elif def_id == "D07_Mixup":
            from Defense.base import DefenseInput
            from Defense.mixup import MixupDefense

            defense = MixupDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.defense_config = config
            common.set_global_seed(seed)
            defense.fit(defense_input)
            model = defense.defended_model

        elif def_id == "D08_LabelSmoothing":
            from Defense.base import DefenseInput
            from Defense.label_smoothing import LabelSmoothingDefense

            defense = LabelSmoothingDefense()
            defense_input = _training_input(DefenseInput, manifest, tensors,
                                            resnet18_cifar_factory)
            defense_input.defense_config = config
            defense.fit(defense_input)
            model = defense.defended_model

        else:
            raise ValueError(f"unknown defense {def_id}")

        # ---- persist checkpoint + STEP-3-style output cache ----------------
        # wrapper predictors (HAMP modify_output, MemGuard) are pickled whole:
        # their state_dict does not match a bare ResNet18
        is_wrapper = type(model).__name__ in ("_HAMPProtectedPredictor", "_MemGuardProtectedPredictor")
        ckpt_path = common.checkpoint_dir("defenses") / f"{def_id}_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(),
                    **({"full_module": model} if is_wrapper else {}),
                    "defense": def_id, "seed": seed,
                    "recipe": RECIPE, "deviation": DEFENSE_DEVIATIONS[def_id]}, ckpt_path)
        summary = cache_partition_outputs(model, seed, def_id, device, tensors)
        record = {
            "seed": seed,
            "defense": def_id,
            "deviation": DEFENSE_DEVIATIONS[def_id],
            "test_acc": summary["test_acc"],
            "partition_acc": summary,
            "defense_seconds": time.time() - t0,
            "git_commit": common.git_commit(),
            "versions": common.env_versions(),
        }
        if def_id == "D04_EarlyStop" and getattr(defense, "selected_epoch", None) is not None:
            record["selected_epoch"] = defense.selected_epoch
            record["stopped_epoch"] = defense.stopped_epoch
            record["stop_reason"] = defense.stop_reason
        if def_id == "D06_MemGuard":
            record["memguard_stats"] = {
                "perturbed_fraction": output.metadata.get("perturbed_fraction"),
                "argmax_preserved_fraction": output.metadata.get("argmax_preserved_fraction"),
                "surrogate_train_accuracy": output.metadata.get("surrogate_train_accuracy"),
            }
        common.write_json(common.logs_dir() / f"defense_{def_id}_seed{seed}.json", record)
        common.mark_done("defense", tag)
        print(f"[done] {tag}: test_acc={summary['test_acc']:.4f} "
              f"in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--defenses", nargs="*", default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override recipe epochs (smoke test only)")
    args = parser.parse_args()
    run(args.seed, args.defenses or DEFENSE_ORDER, args.epochs)
