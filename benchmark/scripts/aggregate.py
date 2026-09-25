"""STEP 11-12 (plan §15, W12): aggregate into the four main tables (§20).

Table 1  Target Utility        Dataset × Model → Clean Accuracy
Table 2  Clean Attack          Dataset × Attack → AUROC / Attack Acc / TPR@1% / 0.1% / 0%
Table 3  Defense Utility       Dataset × Defense → Clean Acc / Defended Acc / Utility Drop
Table 4  Defense Privacy       Dataset × Defense × Attack → Clean AUROC /
                               Defended AUROC / Privacy Gain

3 seeds → mean ± std; CSV + JSON under results/summary/ (deliverables §4).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from benchmark.bmk import common

METRIC_KEYS = ["auroc", "attack_accuracy", "tpr_at_1pct_fpr", "tpr_at_0_1pct_fpr", "tpr_at_0pct_fpr"]


def _load(path: Path):
    return common.read_json(path)


def _mean_std(values):
    arr = [v for v in values if v is not None]
    if not arr:
        return None, None, 0
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0, len(arr)


def collect_clean_attacks(seeds):
    rows = {}
    for seed in seeds:
        for path in sorted((common.results_dir("clean_attack") / f"seed{seed}").glob("*.json")):
            rec = _load(path)
            key = rec["attack"]
            rows.setdefault(key, []).append(rec)
    return rows


def collect_defense_attacks(seeds):
    rows = {}
    for seed in seeds:
        base = common.results_dir("defense_attack")
        for path in sorted(base.glob("*/seed%d/*.json" % seed)):
            rec = _load(path)
            rows.setdefault((rec["defense"], rec["attack"]), []).append(rec)
    return rows


def collect_utility(seeds):
    """Utility = target_nonmember (= official test) accuracy per model."""
    utility = {"clean": {}}
    for seed in seeds:
        clean_summary = common.cache_dir(seed, "clean") / "summary.json"
        if clean_summary.exists():
            utility["clean"][seed] = _load(clean_summary)["test_acc"]
        seed_cache_dir = common.cache_dir(seed).parent  # .../seed{k}
        if not seed_cache_dir.exists():
            continue
        for def_dir in sorted(seed_cache_dir.iterdir()):
            if def_dir.name == "clean":
                continue
            summary = def_dir / "summary.json"
            if summary.exists():
                utility.setdefault(def_dir.name, {})[seed] = _load(summary)["test_acc"]
    return utility


def fmt(mean, std):
    if mean is None:
        return "N/A"
    return f"{mean:.4f} ± {std:.4f}" if std is not None else f"{mean:.4f}"


def run(seeds) -> dict:
    summary_dir = common.results_dir("summary")
    summary_dir.mkdir(parents=True, exist_ok=True)
    versions = {"git_commit": common.git_commit(), "versions": common.env_versions()}

    utility = collect_utility(seeds)

    # ------------------------------------------------- Table 1: target utility
    table1 = []
    clean_accs = [utility["clean"].get(s) for s in seeds if utility["clean"].get(s) is not None]
    m, sd, n = _mean_std(clean_accs)
    table1.append({"dataset": common.DATASET, "model": "resnet18_cifar",
                   "clean_acc_mean": m, "clean_acc_std": sd, "seeds": n})

    # ------------------------------------------------- Table 2: clean attacks
    clean_rows = collect_clean_attacks(seeds)
    table2 = []
    for attack, recs in clean_rows.items():
        row = {"dataset": common.DATASET, "model": "resnet18_cifar", "attack": attack,
               "family": recs[0]["family"], "seeds": len(recs)}
        for key in METRIC_KEYS:
            m, sd, _ = _mean_std([r.get(key) for r in recs])
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = sd
        table2.append(row)

    # ------------------------------------------------ Table 3: defense utility
    table3 = []
    for def_id in sorted(k for k in utility if k != "clean"):
        def_accs = [utility[def_id].get(s) for s in seeds if utility[def_id].get(s) is not None]
        m, sd, n = _mean_std(def_accs)
        cm, cs, _ = _mean_std(clean_accs)
        table3.append({
            "dataset": common.DATASET, "defense": def_id, "seeds": n,
            "clean_acc_mean": cm, "clean_acc_std": cs,
            "defended_acc_mean": m, "defended_acc_std": sd,
            "utility_drop_mean": (cm - m) if (cm is not None and m is not None) else None,
        })

    # ---------------------------------------------- Table 4: defense privacy
    defense_rows = collect_defense_attacks(seeds)
    clean_by_attack = {a: {r["seed"]: r for r in recs} for a, recs in clean_rows.items()}
    table4 = []
    for (def_id, attack), recs in sorted(defense_rows.items()):
        row = {"dataset": common.DATASET, "defense": def_id, "attack": attack,
               "seeds": len(recs), "deviation": recs[0].get("deviation")}
        for key in METRIC_KEYS:
            defended = [r.get(key) for r in recs]
            clean_vals = [clean_by_attack.get(attack, {}).get(r["seed"], {}).get(key)
                          for r in recs]
            m, sd, _ = _mean_std(defended)
            cm, _, _ = _mean_std(clean_vals)
            row[f"{key}_mean"] = m
            row[f"{key}_std"] = sd
            if key == "auroc":
                row["clean_auroc_mean"] = cm
                row["privacy_gain_mean"] = (cm - m) if (cm is not None and m is not None) else None
        table4.append(row)

    # ---------------------------------------------------------------- persist
    payload = {"tables": {"table1_target_utility": table1,
                          "table2_clean_attack": table2,
                          "table3_defense_utility": table3,
                          "table4_defense_privacy": table4},
               **versions, "seeds": seeds}
    common.write_json(summary_dir / "four_tables.json", payload)

    for name, rows in [("table1_target_utility", table1), ("table2_clean_attack", table2),
                       ("table3_defense_utility", table3), ("table4_defense_privacy", table4)]:
        if not rows:
            continue
        keys = sorted({k for r in rows for k in r})
        with open(summary_dir / f"{name}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    print(f"[aggregate] tables written to {summary_dir}")
    print(f"  Table2 clean attacks: {len(table2)} rows | Table3 defenses: {len(table3)} rows | "
          f"Table4 defense×attack: {len(table4)} rows")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    run(args.seeds)
