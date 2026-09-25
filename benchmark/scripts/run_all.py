"""Resume-safe benchmark pipeline driver (plan §15 STEP 1-12).

Every stage checks its own done-marker, so re-running after an interruption
continues where it stopped. Intended to run inside tmux/nohup on the 4090:

    tmux new -s mia4090
    python -m benchmark.scripts.run_all --seeds 0 1 2 2>&1 | tee -a benchmark/logs/pipeline.log

Smoke mode (--epochs 2) shrinks every training run for the Day-1 timing
check; it must NOT be used for the official results (markers are keyed by
stage name, so a full run after a smoke run redoes everything — delete
benchmark/logs/done/ first or use a fresh workspace).
"""

from __future__ import annotations

import argparse
import time
import traceback

from benchmark.bmk import common
from benchmark.bmk.attacks import ATTACK_ORDER, REATTACK_ORDER
from benchmark.scripts import build_manifest, train_target, build_bundles, run_attacks, \
    train_defenses, reattack, aggregate, check_fairness
from benchmark.scripts.train_defenses import DEFENSE_ORDER

STAGE_ORDER = ["manifest", "target", "bundles", "clean_attacks", "fairness",
               "defenses", "reattack", "aggregate"]


def run_stage(stage: str, seeds, epochs, defenses) -> bool:
    t0 = time.time()
    print(f"\n{'=' * 70}\n[stage] {stage}\n{'=' * 70}", flush=True)
    try:
        if stage == "manifest":
            for seed in seeds:
                build_manifest.build(seed)
        elif stage == "target":
            for seed in seeds:
                train_target.run(seed) if epochs is None else _train_target_epochs(seed, epochs)
        elif stage == "bundles":
            for seed in seeds:
                build_bundles.run(seed, epochs if epochs is not None
                                  else int(__import__("benchmark.bmk.train", fromlist=["RECIPE"]).RECIPE["epochs"]))
        elif stage == "clean_attacks":
            for seed in seeds:
                run_attacks.run(seed, ATTACK_ORDER)
        elif stage == "fairness":
            for seed in seeds:
                check_fairness.run(seed)
        elif stage == "defenses":
            for seed in seeds:
                train_defenses.run(seed, defenses, epochs)
        elif stage == "reattack":
            for seed in seeds:
                reattack.run(seed, defenses, REATTACK_ORDER)
        elif stage == "aggregate":
            aggregate.run(seeds)
        else:
            raise ValueError(stage)
    except Exception:
        traceback.print_exc()
        common.log_to_file(f"stage {stage} FAILED after {time.time() - t0:.0f}s", "pipeline")
        return False
    common.log_to_file(f"stage {stage} done in {(time.time() - t0) / 60:.1f} min", "pipeline")
    return True


def _train_target_epochs(seed: int, epochs: int) -> None:
    """Smoke path: reduced epochs (timing backfill only)."""
    import torch

    tag = f"target_seed{seed}"
    if common.is_done("target", tag):
        print(f"[skip] {tag}")
        return

    from benchmark.bmk.data import gather_global, load_cifar10_tensors
    from benchmark.bmk.models import resnet18_cifar_factory
    from benchmark.bmk.train import cache_partition_outputs, train_model

    device = common.resolve_device()
    manifest = common.load_manifest(seed)
    tensors = load_cifar10_tensors()
    X, y = gather_global(manifest["partitions"]["target_train"]["global_indices"], tensors)
    Xv, yv = gather_global(manifest["partitions"]["target_val"]["global_indices"], tensors)
    model = resnet18_cifar_factory()
    t0 = time.time()
    train_model(model, X, y, seed=seed, device=device, val_X=Xv, val_y=yv,
                epochs=epochs, log_name=f"smoke_target_seed{seed}")
    seconds = time.time() - t0
    ckpt = common.checkpoint_dir("targets") / f"target_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "epochs": epochs}, ckpt)
    summary = cache_partition_outputs(model, seed, "clean", device, tensors)
    common.write_json(common.logs_dir() / f"smoke_target_seed{seed}.json", {
        "seed": seed, "epochs": epochs, "train_seconds": seconds,
        "seconds_per_epoch": seconds / epochs, "test_acc": summary["test_acc"],
        "versions": common.env_versions(),
    })
    common.mark_done("target", f"target_seed{seed}")
    print(f"[smoke] target seed{seed}: {seconds / epochs:.1f}s/epoch, "
          f"test_acc={summary['test_acc']:.4f}")


def main(seeds, stages, epochs, defenses) -> None:
    for stage in STAGE_ORDER:
        if stages and stage not in stages:
            continue
        ok = run_stage(stage, seeds, epochs, defenses)
        if not ok:
            print(f"[abort] stage {stage} failed; fix and re-run (resume-safe)", flush=True)
            raise SystemExit(1)
    print("\n[run_all] benchmark complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--stages", nargs="*", default=None,
                        help="subset of: " + " ".join(STAGE_ORDER))
    parser.add_argument("--epochs", type=int, default=None,
                        help="smoke-test epoch override (timing only)")
    parser.add_argument("--defenses", nargs="*", default=DEFENSE_ORDER)
    args = parser.parse_args()
    main(args.seeds, args.stages, args.epochs, args.defenses)
