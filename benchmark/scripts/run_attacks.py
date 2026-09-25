#!/usr/bin/env python
"""STEP 6 入口：对（clean 或 defended）target 运行攻击集。

用法:
  python benchmark/scripts/run_attacks.py --dataset purchase --seed 0
  python benchmark/scripts/reattack.py 用于防御后（见 reattack 脚本）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab.attack_runner import run_all_attacks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defense", default="clean")
    ap.add_argument("--attacks", default=None, help="逗号分隔；默认取配置全集")
    ap.add_argument("--stage", default=None, help="结果子目录：clean_attack / defense_attack")
    args = ap.parse_args()

    C.ensure_dirs()
    attacks = args.attacks.split(",") if args.attacks else None
    stage = args.stage or ("clean_attack" if args.defense == "clean" else "defense_attack")
    run_all_attacks(args.dataset, args.seed, defense=args.defense, attacks=attacks, stage=stage)


if __name__ == "__main__":
    main()
