"""STEP 6-7 (plan §15): run all 10 attacks on the Clean Target.

Per-attack resume markers; one JSON per (seed, attack) plus a combined
per-seed file and CSV under results/clean_attack/ (deliverables §4:
every record carries the metrics block from Attack/eval_common.py, the
git commit hash and the torch/cuda versions).
"""

from __future__ import annotations

import argparse
import time

from benchmark.bmk import common
from benchmark.bmk.attacks import ATTACK_ORDER, AttackContext, run_one_attack


def run(seed: int, attacks: list[str]) -> None:
    ctx = AttackContext(seed, "clean", load_raw=True)  # raw images for QMIA
    env = {"git_commit": common.git_commit(), "versions": common.env_versions(),
           "protocol": "plan v2 (2026-09) §8/§14.1; 4090 side"}
    out_dir = common.results_dir("clean_attack") / f"seed{seed}"
    for name in attacks:
        tag = f"clean_clean_{name}_seed{seed}"
        record_path = out_dir / f"{name}.json"
        if common.is_done("attack", tag) and record_path.exists():
            print(f"[skip] {tag}")
            continue
        t0 = time.time()
        record = run_one_attack(ctx, name, {"stage": "clean", "threat_model": "non-adaptive"})
        record["attack_seconds"] = time.time() - t0
        record.update(env)
        common.write_json(record_path, record)
        common.mark_done("attack", tag)
        print(f"[attack] {name:24s} AUROC={record['auroc']:.4f} "
              f"Acc={record['attack_accuracy'] if record['attack_accuracy'] is None else round(record['attack_accuracy'], 4)} "
              f"TPR@1%={record['tpr_at_1pct_fpr']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attacks", nargs="*", default=None)
    args = parser.parse_args()
    run(args.seed, args.attacks or ATTACK_ORDER)
