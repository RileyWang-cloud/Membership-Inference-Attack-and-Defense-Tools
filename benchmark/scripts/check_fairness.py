#!/usr/bin/env python
"""W14：方案 §21 公平性清单的自动校验。

用法: python benchmark/scripts/check_fairness.py [--datasets purchase,texas,mnist]
退出码非 0 表示存在 FAIL 项。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C
from benchmark.miab.defense_runner import DEFENSE_IDS
from benchmark.miab.manifest import validate_manifest

CHECKS: list = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def _manifests(dataset):
    out = {}
    for seed_dir in sorted((C.SPLITS_DIR / dataset).glob("seed*")):
        seed = int(seed_dir.name.replace("seed", ""))
        with open(seed_dir / "manifest.json") as f:
            out[seed] = json.load(f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(d for d in C.DATASET_META))
    args = ap.parse_args()
    datasets = args.datasets.split(",")
    failures = []

    for ds in datasets:
        mdir = C.SPLITS_DIR / ds
        if not mdir.exists():
            print(f"[SKIP] {ds}: no manifests")
            continue
        manifests = _manifests(ds)
        print(f"\n=== {ds} (seeds {sorted(manifests)}) ===")
        is_image = C.DATASET_META[ds]["is_image"]

        # 1) manifest 结构校验（§4 全部不变量）
        try:
            for seed, m in manifests.items():
                validate_manifest(m)
            print("[PASS] manifest 不变量（分区互斥、member_eval⊂train、pool/population 关系）")
        except AssertionError as e:
            failures.append(f"{ds}: manifest invalid: {e}")
            print(f"[FAIL] manifest invalid: {e}")
            continue

        for seed, m in manifests.items():
            parts = {k: np.asarray(v["indices"]) for k, v in m["partitions"].items()}
            tag = f"{ds}/s{seed}"

            # 2) 派生种子规则（§3）
            exp_shadow = 100 * seed + 50
            exp_refs = [100 * seed + i for i in range(len(m["derived_seeds"]["reference_models"]))]
            ok = (m["derived_seeds"]["shadow_model"] == exp_shadow
                  and m["derived_seeds"]["reference_models"] == exp_refs)
            print(f"[{'PASS' if ok else 'FAIL'}] {tag} 派生种子规则 shadow={exp_shadow} refs={exp_refs}")
            if not ok:
                failures.append(f"{tag}: derived seeds wrong")

            # 3) Reference bundle：in/out 无偏 50% 采样
            rb = C.BUNDLE_DIR / ds / f"seed{seed}" / "reference_bundle.npz"
            if rb.exists():
                z = np.load(rb)
                row_sums = z["in_out_matrix"].sum(axis=1)
                pool_n = z["in_out_matrix"].shape[1]
                ratio = row_sums / pool_n
                ok = np.allclose(ratio, 0.5, atol=0.01) and z["ref_probs_pool"].shape[0] == 4
                print(f"[{'PASS' if ok else 'FAIL'}] {tag} ref 50% 无偏采样（subset_ratio={ratio.round(3)}）")
                if not ok:
                    failures.append(f"{tag}: ref subset ratio not 0.5")

            # 4) target checkpoint 唯一 + 全分区缓存尺寸一致
            ckpt = C.CHECKPOINT_DIR / "targets" / ds / f"target_seed{seed}.pt"
            from benchmark.miab.caching import CACHED_PARTITIONS
            cache_ok = ckpt.exists()
            # §7 缓存清单：partitions + rmia_population（顶层键）的期望尺寸
            check_sizes = {k: v["size"] for k, v in m["partitions"].items()}
            check_sizes["rmia_population"] = m["rmia_population"]["size"]
            for pname in CACHED_PARTITIONS:
                f = C.CACHE_DIR / ds / f"seed{seed}" / "clean" / f"{pname}_probs.npy"
                if not f.exists():
                    cache_ok = False
                    break
                arr = np.load(f)
                if arr.shape[0] != check_sizes[pname]:
                    cache_ok = False
                    break
            print(f"[{'PASS' if cache_ok else 'FAIL'}] {tag} 单一 target checkpoint + 缓存尺寸=manifest")
            if not cache_ok:
                failures.append(f"{tag}: cache/checkpoint mismatch")

            # 5) 防御侧：相同评估样本 + recipe/deviation 留档
            da = C.RESULTS_DIR / "defense_attack" / ds / f"seed{seed}"
            if da.exists():
                for did in DEFENSE_IDS:
                    rec_f = da / f"defense_{did}.json"
                    if rec_f.exists():
                        rec = json.loads(rec_f.read_text())
                        if "recipe" not in rec or "deviation" not in rec:
                            failures.append(f"{tag}/{did}: missing recipe/deviation")
                            print(f"[FAIL] {tag}/{did} 缺 recipe/deviation 记录")

            # 6) 表格数据集 TPR@0.1% 必须为 N/A；图像必须有值
            ca = C.RESULTS_DIR / "clean_attack" / ds / f"seed{seed}" / "attacks_clean.json"
            if ca.exists():
                rows = json.loads(ca.read_text())
                ok = True
                for r in rows:
                    v = r.get("tpr_at_0_1pct_fpr")
                    if is_image and v is None:
                        ok = False
                    if not is_image and v is not None:
                        ok = False
                print(f"[{'PASS' if ok else 'FAIL'}] {tag} TPR@0.1%FPR 口径（图像=值 / 表格=N/A）")
                if not ok:
                    failures.append(f"{tag}: tpr@0.1% convention violated")

    print("\n==== 总体 ====")
    if failures:
        for f_ in failures:
            print("[FAIL]", f_)
        sys.exit(1)
    print("[PASS] 全部自动校验通过（§21 其余条目为结构性保证：阈值仅 shadow 校准、"
          "MemGuard 仅用 Shadow Bundle、防御共用 target_train——见实现与 deviation 记录）")


if __name__ == "__main__":
    main()
