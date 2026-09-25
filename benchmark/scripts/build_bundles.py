"""STEP 4-5 (plan §15): shared Shadow Bundle + Reference Bundle (plan §9, W7/W8).

Shadow: exactly 1 model per (dataset, seed), seed = 100*seed+50, trained on
shadow_train (10k) with the §5.3 recipe.
References: exactly 4 models, seed_i = 100*seed+i, each trained on a 50%
random subset of reference_pool (12,500) — identical recipe to the target
(plan §4.4). Everything downstream hydrates from these bundles; no attack
ever retrains attack-side resources (plan §9.3, §13).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from benchmark.bmk import common
from benchmark.bmk.data import gather_global, load_cifar10_tensors
from benchmark.bmk.models import resnet18_cifar_factory
from benchmark.bmk.train import RECIPE, evaluate_accuracy, make_loader, train_model_with_retry


@torch.no_grad()
def _probabilities(model: nn.Module, X: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start : start + batch_size].to(device))
        chunks.append(torch.softmax(logits.float().cpu(), dim=1).numpy())
    return np.concatenate(chunks)


@torch.no_grad()
def _losses(model: nn.Module, X: torch.Tensor, y: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    model.eval()
    chunks = []
    for start in range(0, len(X), batch_size):
        logits = model(X[start : start + batch_size].to(device)).float().cpu()
        chunks.append(nn.functional.cross_entropy(logits, y[start : start + batch_size], reduction="none").numpy())
    return np.concatenate(chunks)


def run(seed: int, epochs: int) -> None:
    tag = f"bundles_seed{seed}"
    if common.is_done("bundles", tag):
        print(f"[skip] {tag} already done")
        return
    device = common.resolve_device()
    manifest = common.load_manifest(seed)
    tensors = load_cifar10_tensors()
    parts = manifest["partitions"]
    out_dir = common.bundle_dir(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = manifest["derived_seeds"]

    # ---------------------------------------------------------- shadow (W7)
    X_shadow, y_shadow = gather_global(parts["shadow_train"]["global_indices"], tensors)
    t0 = time.time()
    shadow, _, shadow_actual_seed, shadow_retries = train_model_with_retry(
        resnet18_cifar_factory, seed=seeds["shadow_seed"], X=X_shadow, y=y_shadow,
        device=device, epochs=epochs, log_name=f"shadow_seed{seed}")
    shadow_seconds = time.time() - t0
    shadow_ckpt = common.checkpoint_dir("shadows") / f"shadow_seed{seed}.pt"
    torch.save({"state_dict": shadow.state_dict(), "seed": shadow_actual_seed,
                "nominal_seed": seeds["shadow_seed"],
                "divergence_retries": shadow_retries, "recipe": RECIPE}, shadow_ckpt)

    member_outputs = _probabilities(shadow, X_shadow, device)
    member_losses = _losses(shadow, X_shadow, y_shadow, device)
    X_shadow_test, y_shadow_test = gather_global(parts["shadow_test"]["global_indices"], tensors)
    nonmember_outputs = _probabilities(shadow, X_shadow_test, device)
    nonmember_losses = _losses(shadow, X_shadow_test, y_shadow_test, device)
    # shadow outputs on the RMIA population (needed for shadow-side RMIA
    # calibration, §14.1: the shadow model plays "target" on population z too)
    pop_global = np.asarray(manifest["rmia_population"]["global_indices"], dtype=np.int64)
    X_pop, y_pop = gather_global(pop_global, tensors)
    population_outputs = _probabilities(shadow, X_pop, device)
    np.savez(
        out_dir / "shadow_bundle.npz",
        shadow_member_outputs=member_outputs.astype(np.float32),
        shadow_member_labels=y_shadow.numpy(),
        shadow_member_losses=member_losses,
        shadow_nonmember_outputs=nonmember_outputs.astype(np.float32),
        shadow_nonmember_labels=y_shadow_test.numpy(),
        shadow_nonmember_losses=nonmember_losses,
        shadow_population_outputs=population_outputs.astype(np.float32),
        shadow_population_labels=y_pop.numpy(),
        shadow_seed=np.int64(shadow_actual_seed),
        shadow_nominal_seed=np.int64(seeds["shadow_seed"]),
    )
    print(f"[bundle] shadow done in {shadow_seconds / 60:.1f} min "
          f"(train acc on members: {(member_outputs.argmax(1) == y_shadow.numpy()).mean():.4f})")

    # ----------------------------------------------------- references (W8)
    pool_global = np.asarray(manifest["reference_pool"]["pool_to_global"], dtype=np.int64)
    X_pool, y_pool = gather_global(pool_global, tensors)
    eval_global = np.concatenate(
        [parts["target_member_eval"]["global_indices"], parts["target_nonmember"]["global_indices"]]
    )
    X_eval, y_eval = gather_global(eval_global, tensors)
    pop_pool_pos = np.asarray(manifest["rmia_population"]["pool_positions"], dtype=np.int64)

    ref_probs_eval, ref_probs_pool, ref_probs_pop = [], [], []
    ref_actual_seeds, ref_retries = [], []
    in_out_matrix = np.zeros((common.NUM_REFS, len(pool_global)), dtype=bool)
    ref_seconds = []
    for i in range(common.NUM_REFS):
        subset_pool_pos = np.asarray(manifest["reference_pool"]["subsets_pool_pos"][i], dtype=np.int64)
        in_out_matrix[i, subset_pool_pos] = True
        t0 = time.time()
        model, _, ref_seed_actual, retries = train_model_with_retry(
            resnet18_cifar_factory, seed=seeds["reference_seeds"][i],
            X=X_pool[subset_pool_pos], y=y_pool[subset_pool_pos],
            device=device, epochs=epochs, log_name=f"ref{i}_seed{seed}")
        ref_seconds.append(time.time() - t0)
        ref_actual_seeds.append(ref_seed_actual)
        ref_retries.append(retries)
        ref_ckpt = common.checkpoint_dir("references") / f"ref_{i}_seed{seed}.pt"
        torch.save({"state_dict": model.state_dict(), "seed": ref_seed_actual,
                    "nominal_seed": seeds["reference_seeds"][i],
                    "divergence_retries": retries,
                    "subset_pool_pos": subset_pool_pos, "recipe": RECIPE}, ref_ckpt)
        ref_probs_eval.append(_probabilities(model, X_eval, device).astype(np.float32))
        ref_probs_pool.append(_probabilities(model, X_pool, device).astype(np.float32))
        ref_probs_pop.append(ref_probs_pool[-1][pop_pool_pos])
        del model
        torch.cuda.empty_cache()

    np.savez(
        out_dir / "reference_bundle.npz",
        # eval order: member_eval (10k) then nonmember (10k) — fixed for all attacks
        ref_probabilities=np.stack(ref_probs_eval),           # [4, 20000, 10]
        ref_pool_probabilities=np.stack(ref_probs_pool),      # [4, 25000, 10]
        ref_population_probabilities=np.stack(ref_probs_pop), # [4, 2500, 10]
        ref_in_out_matrix=in_out_matrix,                      # [4, 25000]
        eval_labels=y_eval.numpy(),
        pool_labels=y_pool.numpy(),
        pool_to_global=pool_global,
        population_pool_pos=pop_pool_pos,
        reference_seeds=np.asarray(seeds["reference_seeds"]),
        reference_actual_seeds=np.asarray(ref_actual_seeds),
        eval_global_indices=eval_global,
    )
    record = {
        "seed": seed,
        "shadow": {"seed": shadow_actual_seed, "nominal_seed": seeds["shadow_seed"],
                   "divergence_retries": shadow_retries,
                   "train_seconds": shadow_seconds, "train_size": len(X_shadow)},
        "references": [
            {"index": i, "seed": ref_actual_seeds[i], "nominal_seed": seeds["reference_seeds"][i],
             "divergence_retries": ref_retries[i], "train_seconds": ref_seconds[i],
             "subset_size": int(in_out_matrix[i].sum())}
            for i in range(common.NUM_REFS)
        ],
        "epochs": epochs,
    }
    common.write_json(out_dir / "bundle_stats.json", record)
    common.mark_done("bundles", tag)
    print(f"[done] bundles for seed {seed} at {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=int(RECIPE["epochs"]))
    args = parser.parse_args()
    run(args.seed, args.epochs)
