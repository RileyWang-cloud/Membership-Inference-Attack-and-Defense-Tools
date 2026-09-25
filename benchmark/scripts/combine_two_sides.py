#!/usr/bin/env python
"""双机合并：3090 侧（purchase/texas/mnist）+ 4090 侧（cifar10）→ 完整四张主表。

- 3090 侧：读 benchmark/results/summary/*.csv（m±s 字符串格式）
- 4090 侧：从 git 分支 benchmark/cifar10-resnet18-4090 读 *_mean/_std 长格式
- 输出：benchmark/results/summary_combined/table{1..4}.csv + summary_combined.md
"""

from __future__ import annotations

import csv
import io
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C

BR = "origin/benchmark/cifar10-resnet18-4090"

DEFENSE_MAP = {  # 统一防御命名
    "DPSGD": "D01_DP_SGD", "RelaxLoss": "D02_RelaxLoss", "HAMP": "D03_HAMP",
    "EarlyStop": "D04_EarlyStop", "AdvReg": "D05_AdvReg", "MemGuard": "D06_MemGuard",
    "Mixup": "D07_Mixup", "LabelSmoothing": "D08_LabelSmoothing",
}
DATASET_ORDER = {"purchase": 0, "texas": 1, "mnist": 2, "cifar10": 3}


def git_show(path: str) -> str:
    return subprocess.check_output(["git", "show", f"{BR}:{path}"], cwd=str(C.REPO_ROOT)).decode()


def git_rows(path: str) -> list:
    return list(csv.DictReader(io.StringIO(git_show(path))))


def parse_ms(cell: str):
    if cell in ("N/A", "", None):
        return None, None
    m = re.match(r"([-\d.]+)±([-\d.]+)", str(cell))
    if m:
        return float(m.group(1)), float(m.group(2))
    return float(cell), 0.0


def fmt(mean, std) -> str:
    if mean is None:
        return "N/A"
    return f"{mean:.4f}±{std:.4f}" if std else f"{mean:.4f}"


def main() -> None:
    out = C.RESULTS_DIR / "summary_combined"
    out.mkdir(parents=True, exist_ok=True)
    mine = C.RESULTS_DIR / "summary"

    # ---------- Table 1 ----------
    t1 = {}
    with open(mine / "table1_target_utility.csv") as f:
        for r in csv.DictReader(f):
            mu, sd = parse_ms(r["clean_test_acc(mean±std)"])
            t1[r["dataset"]] = (mu, sd)
    for r in git_rows("benchmark/results/summary/table1_target_utility.csv"):
        t1["cifar10"] = (float(r["clean_acc_mean"]), float(r["clean_acc_std"]))
    with open(out / "table1_target_utility.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seeds", "clean_test_acc"])
        for ds in sorted(t1, key=lambda d: DATASET_ORDER.get(d, 9)):
            w.writerow([ds, 3, fmt(*t1[ds])])

    # ---------- Table 2 ----------
    t2 = {}
    with open(mine / "table2_clean_attack.csv") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["attack"])
            t2[key] = {
                "family": r["family"],
                "auroc": parse_ms(r["auroc(m±s)"]), "acc": parse_ms(r["accuracy(m±s)"]),
                "tpr1": parse_ms(r["tpr_at_1pct_fpr(m±s)"]),
                "tpr01": parse_ms(r["tpr_at_0_1pct_fpr(m±s)"]),
                "tpr0": parse_ms(r["tpr_at_0pct_fpr(m±s)"]),
            }
    for r in git_rows("benchmark/results/summary/table2_clean_attack.csv"):
        t2[("cifar10", r["attack"])] = {
            "family": r["family"],
            "auroc": (float(r["auroc_mean"]), float(r["auroc_std"])),
            "acc": (float(r["attack_accuracy_mean"]), float(r["attack_accuracy_std"])),
            "tpr1": (float(r["tpr_at_1pct_fpr_mean"]), float(r["tpr_at_1pct_fpr_std"])),
            "tpr01": (float(r["tpr_at_0_1pct_fpr_mean"]), float(r["tpr_at_0_1pct_fpr_std"])),
            "tpr0": (float(r["tpr_at_0pct_fpr_mean"]), float(r["tpr_at_0pct_fpr_std"])),
        }
    with open(out / "table2_clean_attack.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "attack", "family", "auroc", "attack_accuracy",
                    "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr", "tpr_at_0pct_fpr"])
        for (ds, atk) in sorted(t2, key=lambda k: (DATASET_ORDER.get(k[0], 9), k[1])):
            v = t2[(ds, atk)]
            w.writerow([ds, atk, v["family"], fmt(*v["auroc"]), fmt(*v["acc"]),
                        fmt(*v["tpr1"]), fmt(*v["tpr01"]), fmt(*v["tpr0"])])

    # ---------- Table 3 ----------
    t3 = {}
    with open(mine / "table3_defense_utility.csv") as f:
        for r in csv.DictReader(f):
            did = DEFENSE_MAP[r["defense"]]
            t3[(r["dataset"], did)] = tuple(parse_ms(r[k])[0] for k in
                                            ("clean_acc", "defended_acc", "utility_drop"))
    for r in git_rows("benchmark/results/summary/table3_defense_utility.csv"):
        t3[("cifar10", r["defense"])] = (float(r["clean_acc_mean"]),
                                         float(r["defended_acc_mean"]),
                                         float(r["utility_drop_mean"]))
    with open(out / "table3_defense_utility.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "defense", "clean_acc", "defended_acc", "utility_drop"])
        for (ds, did) in sorted(t3, key=lambda k: (DATASET_ORDER.get(k[0], 9), k[1])):
            c, d, u = t3[(ds, did)]
            w.writerow([ds, did, f"{c:.4f}", f"{d:.4f}", f"{u:+.4f}"])

    # ---------- Table 4 ----------
    t4 = {}
    with open(mine / "table4_defense_privacy.csv") as f:
        for r in csv.DictReader(f):
            did = DEFENSE_MAP[r["defense"]]
            t4[(r["dataset"], did, r["attack"])] = tuple(parse_ms(r[k])[0] for k in
                                                         ("clean_auroc", "defended_auroc", "privacy_gain"))
    for r in git_rows("benchmark/results/summary/table4_defense_privacy.csv"):
        t4[("cifar10", r["defense"], r["attack"])] = (float(r["clean_auroc_mean"]),
                                                      float(r["auroc_mean"]),
                                                      float(r["privacy_gain_mean"]))
    with open(out / "table4_defense_privacy.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "defense", "attack", "clean_auroc", "defended_auroc", "privacy_gain"])
        for (ds, did, atk) in sorted(t4, key=lambda k: (DATASET_ORDER.get(k[0], 9), k[1], k[2])):
            c, d, g = t4[(ds, did, atk)]
            w.writerow([ds, did, atk, f"{c:.4f}", f"{d:.4f}", f"{g:+.4f}"])

    # ---------- markdown ----------
    md = ["# 完整 Benchmark 汇总（双机合并：3090 purchase/texas/mnist + 4090 cifar10）",
          "", "3 seeds mean±std；来源：本机 summary/ + 分支 benchmark/cifar10-resnet18-4090 (81f2736)。", ""]
    for name, title in [
        ("table1_target_utility", "Table 1 Target Utility"),
        ("table2_clean_attack", "Table 2 Clean Attack"),
        ("table3_defense_utility", "Table 3 Defense Utility"),
        ("table4_defense_privacy", "Table 4 Defense Privacy"),
    ]:
        md.append(f"## {title}\n")
        md.append("```csv\n" + (out / f"{name}.csv").read_text() + "```\n")
    (out / "summary_combined.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[combine] wrote {out}/table1..4.csv + summary_combined.md "
          f"(T2 rows={len(t2)}, T3 rows={len(t3)}, T4 rows={len(t4)})")


if __name__ == "__main__":
    main()
