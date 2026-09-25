"""W9：防御配置驱动训练器（方案 §10）。

规则（§10.2）：
- Training-time 防御的 optimizer / epochs / batch_size / lr 一律沿用该数据集的
  Clean Target 配方（configs/<name>.yaml 的 model 段），经 defense_config 覆盖；
- 算法自有超参保持防御实现默认值；
- 偏离 Clean 配方的部分记录 deviation 字段；
- 所有防御使用完全相同的 target_train / target_val / 评估样本（paired comparison）。

MemGuard（D06）按 §10.3：噪声生成器只用 Shadow Bundle（shadow_train/shadow_test
的后验），包装 Clean Target，不触碰任何 target 分区。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import config as C
from . import data as D
from . import models as MO
from . import training as T
from .bundles import load_shadow_bundle
from .caching import cache_partition_outputs
from .training import forward_logits, set_global_seed

from Defense.adv_reg import AdvRegDefense
from Defense.base import DefenseInput
from Defense.dp_sgd import DPSGDDefense
from Defense.early_stop import EarlyStopDefense
from Defense.hamp import HAMPDefense
from Defense.label_smoothing import LabelSmoothingDefense
from Defense.memguard import MemGuardDefense
from Defense.mixup import MixupDefense
from Defense.relax_loss import RelaxLossDefense

class _ModelFactory:
    """模块级可 pickle 的 model factory（HAMP predictor 持有引用时仍可序列化）。"""

    def __init__(self, arch: str, input_dim: int, num_classes: int):
        self.arch, self.input_dim, self.num_classes = arch, input_dim, num_classes

    def __call__(self):
        return MO.build_model(self.arch, input_dim=self.input_dim, num_classes=self.num_classes)


DEFENSE_IDS = ["DPSGD", "RelaxLoss", "HAMP", "EarlyStop", "AdvReg", "MemGuard", "Mixup", "LabelSmoothing"]


def _clean_recipe(cfg: Dict[str, Any]) -> Dict[str, Any]:
    m = dict(cfg["model"])
    return {
        "optimizer": ("adam" if m["optimizer"].lower() == "adam" else "sgd"),
        "learning_rate": float(m["lr"]),
        "momentum": float(m.get("momentum", 0.0)),
        "weight_decay": float(m.get("weight_decay", 0.0)),
        "batch_size": int(m["batch_size"]),
        "epochs": int(m["epochs"]),
        "milestones": list(m.get("scheduler", {}).get("milestones", [])),
        "gamma": float(m.get("scheduler", {}).get("gamma", 0.1)),
    }


def run_defense(dataset: str, seed: int, defense_id: str, epochs_override: int = None) -> Dict[str, Any]:
    assert defense_id in DEFENSE_IDS
    cfg = C.load_config(dataset)
    manifest = C.load_manifest(dataset, seed)
    data = D.load_dataset(dataset)
    parts = D.partitions_from_manifest(data, manifest)
    meta = C.DATASET_META[dataset]
    input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))
    recipe = _clean_recipe(cfg)
    if epochs_override is not None:
        recipe["epochs"] = epochs_override

    model_factory = _ModelFactory(cfg["model"]["arch"], input_dim, meta["num_classes"])

    X_tr, y_tr = parts["target_train"]
    X_val, y_val = parts["target_val"]
    X_me, y_me = parts["target_member_eval"]
    X_nm, y_nm = parts["target_nonmember"]

    t0 = time.time()
    deviation = []

    if defense_id == "DPSGD":
        set_global_seed(seed)
        defense = DPSGDDefense(
            batch_size=recipe["batch_size"], epochs=recipe["epochs"],
            learning_rate=recipe["learning_rate"],
            # σ 在 target_val 上选择（purchase seed0 扫描：0.1→0.60 / 0.5→0.29 / 1.0→0.18），
            # 取最大非退化噪声 σ=0.1、C=1.0，全数据集统一；不触碰任何评估集
            noise_multiplier=0.1, max_grad_norm=1.0,
        )
        defense.fit(DefenseInput(None, model_factory, X_tr, y_tr, X_val, y_val))
        defended = defense.defended_model
        deviation.append("dp_sgd_per_sample_autograd_no_accounting_sigma0.1_tuned_on_val")

    elif defense_id == "RelaxLoss":
        set_global_seed(seed)
        defense = RelaxLossDefense()
        defense.fit(DefenseInput(None, model_factory, X_tr, y_tr, X_val, y_val, defense_config={
            **recipe, "alpha": 1.0, "upper": 1.0,
        }))
        defended = defense.defended_model

    elif defense_id == "HAMP":
        set_global_seed(seed)
        # HAMP 是 hybrid 防御：defended target = protected_predictor
        # （rank-preserving replacement 的 reference = Shadow Bundle 的 non-member logits）
        shadow = load_shadow_bundle(dataset, seed)
        defense = HAMPDefense()
        defense.fit(DefenseInput(
            None, model_factory, X_tr, y_tr, X_val, y_val,
            auxiliary_data={"reference_logits": shadow["shadow_test_logits"]},
            defense_config={**recipe},  # optimizer 家族/lr/epochs 对齐 Clean 配方
        ))
        defended = defense.protected_predictor or defense.defended_model
        deviation.append("hamp_recipe_aligned_to_clean+shadow_reference")

    elif defense_id == "EarlyStop":
        set_global_seed(seed)
        defense = EarlyStopDefense()
        defense.fit(DefenseInput(None, model_factory, X_tr, y_tr, X_val, y_val, defense_config={
            **recipe, "patience": 5, "monitor": "val_loss", "restore_best": True,
        }))
        defended = defense.defended_model

    elif defense_id == "AdvReg":
        set_global_seed(seed)
        defense = AdvRegDefense()
        defense.fit(DefenseInput(
            None, model_factory, X_tr, y_tr, X_val, y_val,
            auxiliary_data={"nonmember_data": parts["auxiliary"][0],
                            "nonmember_labels": parts["auxiliary"][1]},
            defense_config={**recipe},
        ))
        defended = defense.defended_model

    elif defense_id == "Mixup":
        set_global_seed(seed)
        defense = MixupDefense()
        defense.fit(DefenseInput(None, model_factory, X_tr, y_tr, X_val, y_val, defense_config={
            **{k: recipe[k] for k in ("batch_size", "epochs", "learning_rate")},
            "alpha": 1.0, "seed": seed,
        }))
        defended = defense.defended_model

    elif defense_id == "LabelSmoothing":
        set_global_seed(seed)
        defense = LabelSmoothingDefense()
        defense.fit(DefenseInput(None, model_factory, X_tr, y_tr, X_val, y_val, defense_config={
            "smoothing": 0.1, "batch_size": recipe["batch_size"],
            "epochs": recipe["epochs"], "learning_rate": recipe["learning_rate"],
        }))
        defended = defense.defended_model
        deviation.append("label_smoothing_internal_adam")

    elif defense_id == "MemGuard":
        shadow = load_shadow_bundle(dataset, seed)
        # MemGuard 的 perturb 路径存在 posteriors/probs 设备错配（cuda+cpu 混用），
        # 本 benchmark 全程 CPU 运行该防御（扰动为轻量优化循环，代价可接受）
        target = model_factory()
        target.load_state_dict(torch.load(
            C.CHECKPOINT_DIR / "targets" / dataset / f"target_seed{seed}.pt",
            map_location="cpu", weights_only=True))
        target = target.to("cpu").eval()

        eval_X = np.concatenate([X_me, X_nm], axis=0)
        eval_y = np.concatenate([y_me, y_nm], axis=0)
        defense = MemGuardDefense(device="cpu")
        defense.fit(DefenseInput(
            target_model=target, samples=eval_X, labels=eval_y,
            auxiliary_data={
                "member_probabilities": shadow["shadow_train_probs"],
                "nonmember_probabilities": shadow["shadow_test_probs"],
            },
        ))
        out = defense.infer(DefenseInput(target_model=target, samples=eval_X, labels=eval_y))
        defended = out.protected_predictor
        deviation.append("memguard_trained_on_shadow_bundle_only")

    else:
        raise ValueError(defense_id)

    elapsed = round(time.time() - t0, 1)

    # Utility（Defended Accuracy on 固定评估集两侧）
    logits_me = forward_logits(defended, X_me)
    logits_nm = forward_logits(defended, X_nm)
    record = {
        "dataset": dataset, "seed": seed, "defense": defense_id,
        "defended_member_eval_acc": float((logits_me.argmax(1) == y_me).mean()),
        "defended_nonmember_acc": float((logits_nm.argmax(1) == y_nm).mean()),
        "deviation": deviation, "defense_seconds": elapsed,
        "recipe": recipe, "git_commit": manifest["generated"]["git_commit"],
    }

    # 全分区输出缓存（供重攻击）
    cache_partition_outputs(defended, parts, defense=defense_id, dataset=dataset, seed=seed)

    # checkpoint（MemGuard 的 protected predictor 直接 pickle 整对象）
    ckpt_dir = C.CHECKPOINT_DIR / "defenses" / dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{defense_id}_seed{seed}.pt"
    if defense_id in ("MemGuard", "HAMP"):
        torch.save(defended, path)
    else:
        torch.save(defended.state_dict(), path)

    out_dir = C.RESULTS_DIR / "defense_attack" / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"defense_{defense_id}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"[defense] {dataset}/s{seed}/{defense_id}: member_acc={record['defended_member_eval_acc']:.4f} "
          f"nonmember_acc={record['defended_nonmember_acc']:.4f} ({elapsed}s)")
    return record
