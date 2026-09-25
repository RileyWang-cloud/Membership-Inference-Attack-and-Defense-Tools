"""STEP 10 (plan §15, W12): re-attack defended targets.

Five representative attacks (plan §12): Loss / Shadow / LiRA / RMIA plus
QMIA stress-test (default-on per §23.5). Shadow-side calibration (mu_m /
mu_nm) is recomputed by the same shadow-as-target path each time — it is a
pure function of the Shadow / Reference bundles, identical across defenses
(non-adaptive threat model, §13).
"""

from __future__ import annotations

import argparse
import time

from benchmark.bmk import common
from benchmark.bmk.attacks import REATTACK_ORDER, AttackContext, run_one_attack
from benchmark.scripts.train_defenses import DEFENSE_DEVIATIONS, DEFENSE_ORDER


def run(seed: int, defenses: list[str], attacks: list[str]) -> None:
    env = {"git_commit": common.git_commit(), "versions": common.env_versions(),
           "protocol": "plan v2 (2026-09) §12/§14.1; 4090 side"}
    for def_id in defenses:
        ctx = AttackContext(seed, def_id, load_raw=True)  # raw images for QMIA
        out_dir = common.results_dir("defense_attack") / def_id / f"seed{seed}"
        for name in attacks:
            tag = f"reattack_{def_id}_{name}_seed{seed}"
            record_path = out_dir / f"{name}.json"
            if common.is_done("reattack", tag) and record_path.exists():
                print(f"[skip] {tag}")
                continue
            t0 = time.time()
            record = run_one_attack(ctx, name, {
                "stage": "reattack",
                "threat_model": "non-adaptive (attack-side resources shared with clean)",
                "deviation": DEFENSE_DEVIATIONS[def_id],
            })
            record["attack_seconds"] = time.time() - t0
            record.update(env)
            common.write_json(record_path, record)
            common.mark_done("reattack", tag)
            print(f"[reattack] {def_id:18s} {name:18s} AUROC={record['auroc']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--defenses", nargs="*", default=DEFENSE_ORDER)
    parser.add_argument("--attacks", nargs="*", default=REATTACK_ORDER)
    args = parser.parse_args()
    run(args.seed, args.defenses, args.attacks)
