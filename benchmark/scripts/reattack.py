#!/usr/bin/env python
"""STEP 10 入口：防御后重攻击（方案 §12：Loss/Shadow/LiRA/RMIA + QMIA）。

用法:
  python benchmark/scripts/reattack.py --dataset purchase --seed 0
  python benchmark/scripts/reattack.py --dataset purchase --seed 0 --defenses DPSGD,Mixup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab.attack_runner import run_all_attacks
from benchmark.miab.defense_runner import DEFENSE_IDS

REATTACK_SET = ["LossAttack", "ShadowBasedAttack", "LiRAAttack", "RMIAAttack", "QMIAAttack"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defenses", default=None)
    args = ap.parse_args()

    C.ensure_dirs()
    ids = args.defenses.split(",") if args.defenses else DEFENSE_IDS
    for did in ids:
        run_all_attacks(args.dataset, args.seed, defense=did,
                        attacks=REATTACK_SET, stage="defense_attack")


if __name__ == "__main__":
    main()
