#!/usr/bin/env python
"""W12b：汇总四张主表（方案 §20），mean ± std over seeds。

本机数据集独立可跑（双机协议：各自输出自己数据集的行）。
输出: results/summary/table{1,2,3,4}.csv + summary.md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark.miab import config as C

METRICS = ["auroc", "accuracy", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr", "tpr_at_0pct_fpr"]


def _load(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _ms(values):
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{np.mean(vals):.4f}±{np.std(vals):.4f}"


def collect():
    datasets = sorted([d.name for d in C.RESULTS_DIR.glob("clean_attack/*") if d.is_dir()])
    clean_rows = []          # (ds, seed, attack, metrics)
    defense_rows = []        # (ds, seed, defense, attack, metrics)
    utility = {}             # (ds, seed) -> {clean: {...}}
    def_utility = {}         # (ds, seed, defense) -> record

    for ds in datasets:
        for seed_dir in sorted((C.RESULTS_DIR / "clean_attack" / ds).glob("seed*")):
            seed = int(seed_dir.name.replace("seed", ""))
            rows = _load(seed_dir / "attacks_clean.json") or []
            for r in rows:
                clean_rows.append((ds, seed, r["attack"], r))
            tlog = _load(C.LOGS_DIR / "targets" / ds / f"target_seed{seed}.json")
            if tlog:
                utility[(ds, seed)] = tlog["metrics"]

        da_dir = C.RESULTS_DIR / "defense_attack" / ds
        if da_dir.exists():
            for seed_dir in sorted(da_dir.glob("seed*")):
                seed = int(seed_dir.name.replace("seed", ""))
                for f in sorted(seed_dir.glob("defense_*.json")):
                    did = f.stem.replace("defense_", "")
                    rec = _load(f)
                    if "attack" in (rec or {}):
                        defense_rows.append((ds, seed, did, rec["attack"], rec))
                    else:
                        def_utility[(ds, seed, did)] = rec
                for f in sorted(seed_dir.glob("attacks_*.json")):
                    did = f.stem.replace("attacks_", "")
                    if did == "clean":
                        continue
                    for r in (_load(f) or []):
                        defense_rows.append((ds, seed, did, r["attack"], r))
    return datasets, clean_rows, defense_rows, utility, def_utility


def main() -> None:
    datasets, clean_rows, defense_rows, utility, def_utility = collect()
    out = C.RESULTS_DIR / "summary"
    out.mkdir(parents=True, exist_ok=True)
    seeds_all = sorted({s for _, s, _, _ in clean_rows})

    # ---- Table 1: Target Utility
    lines = ["dataset,seeds,clean_test_acc(mean±std)"]
    for ds in datasets:
        accs = [utility[(ds, s)]["nonmember_acc"] for s in seeds_all if (ds, s) in utility]
        lines.append(f"{ds},{len(accs)},{_ms(accs)}")
    (out / "table1_target_utility.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- Table 2: Clean Attack
    by_key = defaultdict(list)
    for ds, seed, atk, r in clean_rows:
        by_key[(ds, atk)].append(r)
    lines = ["dataset,attack,family," + ",".join(m + "(m±s)" for m in METRICS)]
    for (ds, atk), rows in sorted(by_key.items()):
        vals = []
        for m in METRICS:
            v = [r.get(m) for r in rows]
            if all(x is None for x in v):
                vals.append("N/A")
            else:
                vals.append(_ms([x for x in v if x is not None]))
        lines.append(f"{ds},{atk},{rows[0]['family']}," + ",".join(vals))
    (out / "table2_clean_attack.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- Table 3: Defense Utility
    lines = ["dataset,defense,seeds,clean_acc,defended_acc,utility_drop"]
    du = defaultdict(list)
    for (ds, seed, did), rec in def_utility.items():
        if rec is None or (ds, seed) not in utility:
            continue
        clean_acc = utility[(ds, seed)]["nonmember_acc"]
        du[(ds, did)].append((clean_acc, rec["defended_nonmember_acc"]))
    for (ds, did), pairs in sorted(du.items()):
        ca = _ms([p[0] for p in pairs])
        da = _ms([p[1] for p in pairs])
        drop = _ms([p[0] - p[1] for p in pairs])
        lines.append(f"{ds},{did},{len(pairs)},{ca},{da},{drop}")
    (out / "table3_defense_utility.csv").write_text("\n".join(lines), encoding="utf-8")

    # ---- Table 4: Defense Privacy
    clean_auroc = {}
    for ds, seed, atk, r in clean_rows:
        clean_auroc[(ds, seed, atk)] = r["auroc"]
    dg = defaultdict(list)
    for ds, seed, did, atk, r in defense_rows:
        ca = clean_auroc.get((ds, seed, atk))
        if ca is None:
            continue
        dg[(ds, did, atk)].append((ca, r["auroc"]))
    lines = ["dataset,defense,attack,clean_auroc,defended_auroc,privacy_gain"]
    for (ds, did, atk), pairs in sorted(dg.items()):
        lines.append(f"{ds},{did},{atk},"
                     f"{_ms([p[0] for p in pairs])},{_ms([p[1] for p in pairs])},"
                     f"{_ms([p[0] - p[1] for p in pairs])}")
    (out / "table4_defense_privacy.csv").write_text("\n".join(lines), encoding="utf-8")

    md = ["# Benchmark 汇总（本机数据集）", ""]
    for i, name in enumerate(["table1_target_utility", "table2_clean_attack",
                              "table3_defense_utility", "table4_defense_privacy"], 1):
        content = (out / f"{name}.csv").read_text(encoding="utf-8")
        md.append(f"## Table {i}: {name}\n")
        md.append("```csv\n" + content + "\n```\n")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[aggregate] wrote {out}/table1..4.csv + summary.md "
          f"({len(clean_rows)} clean rows, {len(defense_rows)} defense rows)")


if __name__ == "__main__":
    main()
