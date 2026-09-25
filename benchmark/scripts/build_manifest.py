"""W1: build the CIFAR-10 split manifest (plan §4.2, §4.4, §3).

Usage: python -m benchmark.scripts.build_manifest --seed 0

The manifest is the single source of truth for every split; no downstream
component may re-split on its own (plan §4.1, §11).
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from benchmark.bmk import common
from benchmark.bmk.data import load_cifar10_tensors


def stratified_permutation(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified shuffle: round-robin over per-class shuffles."""
    by_class = {c: np.where(labels == c)[0] for c in np.unique(labels)}
    for c in by_class:
        rng.shuffle(by_class[c])
    order = []
    max_count = max(len(v) for v in by_class.values())
    for slot in range(max_count):
        for c in sorted(by_class):
            if slot < len(by_class[c]):
                order.append(by_class[c][slot])
    return np.asarray(order, dtype=np.int64)


def build(seed: int) -> None:
    print(f"building manifest for seed={seed}")
    train_X, train_y, test_X, test_y = load_cifar10_tensors()
    train_y_np = train_y.numpy()
    test_y_np = test_y.numpy()
    rng = np.random.default_rng(seed)

    # ---- train-side partitions (50,000 exactly consumed, plan §4.2) ----
    order = stratified_permutation(train_y_np, rng)
    cursor = 0
    partitions: dict[str, np.ndarray] = {}
    for name, size in common.IMAGE_SPLITS.items():
        partitions[name] = np.sort(order[cursor : cursor + size])
        cursor += size
    assert cursor == common.TRAIN_SIZE, f"train side must be fully consumed, got {cursor}"

    # ---- member eval: class-stratified subset of target_train ----
    target_train_idx = partitions["target_train"]
    target_train_labels = train_y_np[target_train_idx]
    rng2 = np.random.default_rng(10_000 + seed)
    per_class = common.MEMBER_EVAL_SIZE // common.NUM_CLASSES
    chosen = []
    for c in range(common.NUM_CLASSES):
        pool = target_train_idx[target_train_labels == c]
        pick = rng2.choice(pool, size=per_class, replace=False)
        chosen.append(pick)
    partitions["target_member_eval"] = np.sort(np.concatenate(chosen))
    assert set(partitions["target_member_eval"]) <= set(partitions["target_train"])

    # ---- nonmember eval: official test set ----
    partitions["target_nonmember"] = np.arange(common.TRAIN_SIZE, common.TRAIN_SIZE + common.TEST_SIZE)

    # ---- reference pool + RMIA population (plan §4.4) ----
    pool_parts = []
    for name in common.REFERENCE_POOL_PARTS:
        # pool indices are stored relative to the pool order (pool_pos -> global)
        pool_parts.append(partitions[name])
    pool_global = np.concatenate(pool_parts)
    # population drawn from the pool, disjoint from member eval by construction
    rng3 = np.random.default_rng(20_000 + seed)
    population_pool_pos = np.sort(rng3.choice(len(pool_global), size=common.POPULATION_SIZE, replace=False))

    seeds = common.derive_seeds(seed)

    # ---- reference subsets: each ref trains on a 50% random subset (§4.4) ----
    ref_subsets_pool_pos = []
    for i in range(common.NUM_REFS):
        r = np.random.default_rng(seeds["reference_seeds"][i])
        subset_size = int(common.REF_SUBSET_RATIO * len(pool_global))
        ref_subsets_pool_pos.append(np.sort(r.choice(len(pool_global), size=subset_size, replace=False)))

    def dist(labels_np, indices, offset=0):
        return {str(c): int(n) for c, n in sorted(Counter(labels_np[indices - offset]).items())}

    manifest = {
        "dataset": common.DATASET,
        "seed": seed,
        "derived_seeds": seeds,
        "global_index_convention": {
            "train": f"[0, {common.TRAIN_SIZE}) -> CIFAR-10 trainset position",
            "test": f"[{common.TRAIN_SIZE}, {common.TRAIN_SIZE + common.TEST_SIZE}) -> official testset position",
        },
        "partitions": {
            name: {
                "size": int(len(idx)),
                "global_indices": idx.tolist(),
                "class_distribution": dist(train_y_np, idx)
                if int(idx.max()) < common.TRAIN_SIZE
                else dist(test_y_np, idx, offset=common.TRAIN_SIZE),
            }
            for name, idx in partitions.items()
        },
        "reference_pool": {
            "size": int(len(pool_global)),
            "pool_to_global": pool_global.tolist(),
            "composed_of": common.REFERENCE_POOL_PARTS,
            "num_models": common.NUM_REFS,
            "subset_ratio": common.REF_SUBSET_RATIO,
            "subset_sizes": [int(len(s)) for s in ref_subsets_pool_pos],
            "subsets_pool_pos": [s.tolist() for s in ref_subsets_pool_pos],
        },
        "rmia_population": {
            "size": int(common.POPULATION_SIZE),
            "pool_positions": population_pool_pos.tolist(),
            "global_indices": pool_global[population_pool_pos].tolist(),
        },
        "dataset_sha256": {
            "train_X": common.sha256_of_array(train_X.numpy()),
            "test_X": common.sha256_of_array(test_X.numpy()),
        },
        "protocol": "plan v2 (2026-09) §3/§4.2/§4.4; 4090 side",
    }

    # integrity checks (plan §4.1: strict disjointness)
    train_side = np.concatenate([partitions[p] for p in common.IMAGE_SPLITS])
    assert len(np.unique(train_side)) == common.TRAIN_SIZE
    pool_set = set(pool_global.tolist())
    eval_set = set(partitions["target_member_eval"].tolist()) | set(partitions["target_nonmember"].tolist())
    assert pool_set.isdisjoint(eval_set), "reference pool must not overlap attack evaluation sets"
    pop_set = set(pool_global[population_pool_pos].tolist())
    assert pop_set <= pool_set and pop_set.isdisjoint(eval_set)
    for subset in ref_subsets_pool_pos:
        assert len(subset) == int(0.5 * len(pool_global))

    common.write_json(common.manifest_path(seed), manifest)
    print(f"manifest written: {common.manifest_path(seed)}")
    print(f"  pool={len(pool_global)} population={common.POPULATION_SIZE} "
          f"member_eval={len(partitions['target_member_eval'])} nonmember={len(partitions['target_nonmember'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--all-seeds", action="store_true", help="build seeds 0,1,2")
    args = parser.parse_args()
    for s in ([0, 1, 2] if args.all_seeds else [args.seed]):
        build(s)
