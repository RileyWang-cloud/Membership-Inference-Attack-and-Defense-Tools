#!/usr/bin/env python
"""W4 入口：训练 Clean Target 并缓存全分区输出（方案 §7 / STEP 2–3）。

用法:
  python benchmark/scripts/train_target.py --dataset purchase --seed 0
  python benchmark/scripts/train_target.py --dataset mnist --seed 0 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from benchmark.miab import config as C
from benchmark.miab import caching as CA
from benchmark.miab import data as D
from benchmark.miab import manifest as M
from benchmark.miab import models as MO
from benchmark.miab import training as T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(C.DATASET_META))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="缩减 epoch 的快速验证")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖配置 epochs")
    args = ap.parse_args()

    C.ensure_dirs()
    cfg = C.load_config(args.dataset)
    manifest = C.load_manifest(args.dataset, args.seed)
    data = D.load_dataset(args.dataset)
    parts = D.partitions_from_manifest(data, manifest)

    meta = C.DATASET_META[args.dataset]
    input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))
    model = MO.build_model(cfg["model"]["arch"], input_dim=input_dim, num_classes=meta["num_classes"])

    epochs = args.epochs or (3 if args.smoke else None)
    X_tr, y_tr = parts["target_train"]
    X_val, y_val = parts["target_val"]
    X_me, y_me = parts["target_member_eval"]
    X_nm, y_nm = parts["target_nonmember"]

    t0 = time.time()
    model, history = T.train_classifier(
        model, X_tr, y_tr, cfg["model"], seed=manifest["derived_seeds"]["split"],
        X_val=X_val, y_val=y_val, epochs_override=epochs,
        log_prefix=f"[{args.dataset}/seed{args.seed}] ",
    )
    elapsed = time.time() - t0

    # Train / Val / Test(=attack eval 两侧) Accuracy
    from benchmark.miab.training import forward_logits
    metrics = {"train_acc": history[-1]["train_acc"], "val_acc": history[-1].get("val_acc")}
    logits_me = forward_logits(model, X_me)
    logits_nm = forward_logits(model, X_nm)
    metrics["member_eval_acc"] = float((logits_me.argmax(1) == y_me).mean())
    metrics["nonmember_acc"] = float((logits_nm.argmax(1) == y_nm).mean())
    metrics["train_seconds"] = round(elapsed, 1)
    metrics["epochs_run"] = len(history)

    ckpt_dir = C.CHECKPOINT_DIR / "targets" / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / f"target_seed{args.seed}.pt")

    log_dir = C.LOGS_DIR / "targets" / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / f"target_seed{args.seed}{'_smoke' if args.smoke else ''}.json", "w") as f:
        json.dump({"history": history, "metrics": metrics, "config": cfg["model"],
                   "git_commit": manifest["generated"]["git_commit"]}, f, indent=1)

    print(f"[target] {args.dataset} seed{args.seed}: {json.dumps(metrics)} ({elapsed:.0f}s)")

    # 全分区输出缓存（§7：clean 防御位）
    CA.cache_partition_outputs(model, parts, defense="clean", dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
