#!/usr/bin/env python
"""W7/W8 入口：训练 Shadow Bundle 与 Reference Bundle（方案 STEP 4–5）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import bundles as B
from benchmark.miab import config as C


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置 epochs（smoke 用）")
    args = ap.parse_args()

    C.ensure_dirs()
    B.build_shadow(args.dataset, args.seed, epochs_override=args.epochs)
    B.build_references(args.dataset, args.seed, epochs_override=args.epochs)


if __name__ == "__main__":
    main()
