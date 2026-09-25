"""STEP 2-3 (plan §15): train the clean target for one seed and cache outputs.

Usage: python -m benchmark.scripts.train_target --seed 0
"""

from __future__ import annotations

import argparse
import time

import torch

from benchmark.bmk import common
from benchmark.bmk.data import gather_global, load_cifar10_tensors
from benchmark.bmk.models import resnet18_cifar_factory
from benchmark.bmk.train import RECIPE, cache_partition_outputs, train_model_with_retry


def run(seed: int) -> None:
    tag = f"target_seed{seed}"
    if common.is_done("target", tag):
        print(f"[skip] {tag} already done")
        return
    device = common.resolve_device()
    manifest = common.load_manifest(seed)
    tensors = load_cifar10_tensors()

    X_train, y_train = gather_global(manifest["partitions"]["target_train"]["global_indices"], tensors)
    X_val, y_val = gather_global(manifest["partitions"]["target_val"]["global_indices"], tensors)

    t0 = time.time()
    model, history, actual_seed, divergence_retries = train_model_with_retry(
        resnet18_cifar_factory, seed=seed, X=X_train, y=y_train,
        device=device, val_X=X_val, val_y=y_val,
        epochs=int(RECIPE["epochs"]), log_name=tag,
    )
    train_seconds = time.time() - t0

    ckpt_path = common.checkpoint_dir("targets") / f"target_seed{seed}.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": actual_seed, "nominal_seed": seed,
                "divergence_retries": divergence_retries, "recipe": RECIPE}, ckpt_path)

    summary = cache_partition_outputs(model, seed, "clean", device, tensors)
    record = {
        "seed": seed,
        "actual_train_seed": actual_seed,
        "divergence_retries": divergence_retries,
        "train_seconds": train_seconds,
        "seconds_per_epoch": train_seconds / int(RECIPE["epochs"]),
        "final_train_acc": history["train_acc"][-1],
        "final_val_acc": history["val_acc"][-1],
        "test_acc": summary["test_acc"],
        "partition_acc": summary,
        "recipe": RECIPE,
        "versions": common.env_versions(),
    }
    common.write_json(common.logs_dir() / f"target_seed{seed}.json", record)
    common.mark_done("target", tag)
    print(f"[done] {tag}: test_acc={summary['test_acc']:.4f} in {train_seconds / 60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.seed)
