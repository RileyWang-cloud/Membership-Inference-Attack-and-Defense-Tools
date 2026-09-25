#!/usr/bin/env python
"""W1 入口：生成 split manifest。

用法:
  python benchmark/scripts/build_manifest.py --dataset purchase --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab import data as D
from benchmark.miab import manifest as M


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    C.ensure_dirs()
    cfg = C.load_config(args.dataset)
    data = D.load_dataset(args.dataset)
    print(f"[data] {args.dataset} loaded: " + (
        f"X {data['X'].shape}" if not data["is_image"] else
        f"train {data['X_train'].shape} / test {data['X_test'].shape}"
    ))

    for seed in [int(s) for s in args.seeds.split(",")]:
        m = M.build_manifest(args.dataset, seed, cfg, data)
        M.save_manifest(m)
        sizes = {k: v["size"] for k, v in m["partitions"].items()}
        print(f"[manifest] {args.dataset} seed{seed}: {sizes} "
              f"pool={m['reference_pool']['size']} pop={m['rmia_population']['size']} "
              f"reserve={len(m['reserve']['indices'])}")


if __name__ == "__main__":
    main()
