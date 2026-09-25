#!/usr/bin/env python
"""全量运行驱动器（可断点续跑）：本机数据集 × 3 seeds 的完整 Benchmark。

跳过规则：结果文件已存在则跳过对应步骤。DPSGD 放在每个 (ds,seed) 的最后。
用法: python benchmark/scripts/run_all.py [--datasets purchase,texas,mnist] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab.defense_runner import DEFENSE_IDS

PY = sys.executable


def sh(cmd: list) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"!! FAILED ({time.time()-t0:.0f}s): {' '.join(cmd)}", flush=True)
        raise SystemExit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="purchase,texas,mnist")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--skip-dpsgd", action="store_true")
    args = ap.parse_args()

    for ds in args.datasets.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            base = C.SPLITS_DIR / ds / f"seed{seed}" / "manifest.json"
            clean_json = C.RESULTS_DIR / "clean_attack" / ds / f"seed{seed}" / "attacks_clean.json"

            if not base.exists():
                sh([PY, "benchmark/scripts/build_manifest.py", "--dataset", ds, "--seeds", str(seed)])
            if not clean_json.exists():
                sh([PY, "benchmark/scripts/train_target.py", "--dataset", ds, "--seed", str(seed)])
                sh([PY, "benchmark/scripts/build_bundles.py", "--dataset", ds, "--seed", str(seed)])
                sh([PY, "benchmark/scripts/run_attacks.py", "--dataset", ds, "--seed", str(seed)])

            fast = [d for d in DEFENSE_IDS if d != "DPSGD"] + (["DPSGD"] if not args.skip_dpsgd else [])
            for did in fast:
                rec = C.RESULTS_DIR / "defense_attack" / ds / f"seed{seed}" / f"defense_{did}.json"
                if not rec.exists():
                    sh([PY, "benchmark/scripts/train_defenses.py", "--dataset", ds,
                        "--seed", str(seed), "--defenses", did])
                atk = C.RESULTS_DIR / "defense_attack" / ds / f"seed{seed}" / f"attacks_{did}.json"
                if not atk.exists():
                    sh([PY, "benchmark/scripts/reattack.py", "--dataset", ds,
                        "--seed", str(seed), "--defenses", did])

    sh([PY, "benchmark/scripts/aggregate.py"])
    sh([PY, "benchmark/scripts/check_fairness.py", "--datasets", args.datasets])
    print("\nALL DONE")


if __name__ == "__main__":
    main()
