#!/usr/bin/env python
"""W9 入口：按配置运行防御（方案 STEP 8–9）。

用法:
  python benchmark/scripts/train_defenses.py --dataset purchase --seed 0
  python benchmark/scripts/train_defenses.py --dataset purchase --seed 0 --defenses MemGuard,Mixup
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab.defense_runner import DEFENSE_IDS, run_defense


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--defenses", default=None, help="逗号分隔；默认全部 D01–D08")
    ap.add_argument("--epochs", type=int, default=None)
    args = ap.parse_args()

    C.ensure_dirs()
    ids = args.defenses.split(",") if args.defenses else DEFENSE_IDS
    for did in ids:
        run_defense(args.dataset, args.seed, did, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
