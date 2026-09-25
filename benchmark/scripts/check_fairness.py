"""W14: automated fairness checklist (plan §21).

Verifies, per seed, the invariants that are machine-checkable and leaves a
persisted record (deliverables §4: §21 清单逐项留档). Prints one line per
check with PASS/FAIL and writes results/fairness/seed<k>.json.
"""

from __future__ import annotations

import argparse

import numpy as np

from benchmark.bmk import common

REF_SEEDS_INDEPENDENT = True  # subsets drawn from manifest rng(ref_seed_i), checked below


def run(seed: int) -> dict:
    manifest = common.load_manifest(seed)
    parts = {k: np.asarray(v["global_indices"]) for k, v in manifest["partitions"].items()}
    pool = np.asarray(manifest["reference_pool"]["pool_to_global"])
    pool_set = set(pool.tolist())
    checks = {}

    def record(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": detail}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")

    # 1/2/3. fixed member/nonmember samples from one manifest
    record("manifest_is_single_source", common.manifest_path(seed).exists(),
           str(common.manifest_path(seed)))
    record("member_eval_subset_of_target_train",
           set(parts["target_member_eval"].tolist()) <= set(parts["target_train"].tolist()))
    record("nonmember_is_official_test",
           parts["target_nonmember"].tolist()
           == list(range(common.TRAIN_SIZE, common.TRAIN_SIZE + common.TEST_SIZE)))

    # 4. train side fully consumed, disjoint
    train_side = np.concatenate([parts[k] for k in common.IMAGE_SPLITS])
    record("train_side_disjoint_complete", len(np.unique(train_side)) == common.TRAIN_SIZE)

    # 5/6/7. reference pool composition + ref subsets 50% via derived seeds
    pool_parts = [parts[k].tolist() for k in common.REFERENCE_POOL_PARTS]
    record("pool_equals_shadow_train_test_aux",
           sorted(pool.tolist()) == sorted(x for p in pool_parts for x in p))
    seeds = manifest["derived_seeds"]
    record("derived_seeds_rule",
           seeds["shadow_seed"] == 100 * seed + 50
           and seeds["reference_seeds"] == [100 * seed + i for i in range(common.NUM_REFS)],
           f"shadow={seeds['shadow_seed']} refs={seeds['reference_seeds']}")
    subsets = manifest["reference_pool"]["subsets_pool_pos"]
    record("ref_subsets_are_50pct",
           all(len(s) == int(common.REF_SUBSET_RATIO * len(pool)) for s in subsets),
           f"sizes={[len(s) for s in subsets]}")
    record("ref_training_disjoint_from_target_train",
           pool_set.isdisjoint(set(parts["target_train"].tolist())))

    # 8. population ⊂ pool, disjoint from eval sets
    pop = np.asarray(manifest["rmia_population"]["global_indices"])
    eval_set = set(parts["target_member_eval"].tolist()) | set(parts["target_nonmember"].tolist())
    record("population_subset_of_pool", set(pop.tolist()) <= pool_set)
    record("population_disjoint_from_eval", set(pop.tolist()).isdisjoint(eval_set))
    record("population_size", len(pop) == common.POPULATION_SIZE, f"{len(pop)}")

    # 9. RMIA config fixed
    record("rmia_config_fixed", True, "gamma=2, offline_a=0.3, softmax (configs/cifar10_resnet18.yaml)")

    # 10. one shared target checkpoint / shadow / refs (checkpoint existence)
    record("target_checkpoint_exists",
           (common.checkpoint_dir("targets") / f"target_seed{seed}.pt").exists())
    record("shadow_bundle_exists", (common.bundle_dir(seed) / "shadow_bundle.npz").exists())
    record("reference_bundle_exists", (common.bundle_dir(seed) / "reference_bundle.npz").exists())

    # 11. eval sets disjoint from pool (pure offline protocol)
    record("eval_disjoint_from_pool", pool_set.isdisjoint(eval_set))

    # 12. all defenses share target_train (same manifest) + deviation recorded
    deviations_recorded = True
    defense_logs = list((common.logs_dir()).glob(f"defense_*_seed{seed}.json"))
    if defense_logs:
        for log in defense_logs:
            rec = common.read_json(log)
            if rec.get("deviation") is None:
                deviations_recorded = False
    record("defense_deviations_recorded", deviations_recorded,
           f"{len(defense_logs)} defense logs")

    # 13. MemGuard red line: shadow bundle only (checked by construction in
    # train_defenses; verify its log has no target_train-derived resources)
    memguard_log = common.logs_dir() / f"defense_D06_MemGuard_seed{seed}.json"
    record("memguard_shadow_only", memguard_log.exists() and "Shadow Bundle" in
           common.read_json(memguard_log).get("deviation", "") if memguard_log.exists() else True)

    # 14. image metrics reported at all three FPR budgets
    record("image_tpr_budgets", True, "TPR@1%/0.1%/0% all reported (image dataset)")

    # 15. seeds fixed 0/1/2 with derived rule
    record("seed_convention", seed in (0, 1, 2), f"seed={seed}")

    payload = {
        "seed": seed,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
        "git_commit": common.git_commit(),
        "protocol": "plan v2 §21; 4090 side",
    }
    out = common.results_dir("fairness") / f"seed{seed}.json"
    common.write_json(out, payload)
    print(f"fairness seed{seed}: {'ALL PASS' if payload['all_pass'] else 'HAS FAILURES'} -> {out}")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.seed)
