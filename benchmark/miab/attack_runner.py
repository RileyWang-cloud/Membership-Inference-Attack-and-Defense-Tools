"""攻击编排层：从缓存 + Bundle 运行 10 种攻击（方案 §8），统一 eval_common 口径。

数据流（非自适应威胁模型，§13）：
- Target 侧：cache/target_outputs/<dataset>/seed<k>/<defense>/ 的 probs/loss/labels
- 攻击侧：Shadow Bundle + Reference Bundle（Clean 与全部 Defense 复用）
- Attack Accuracy 阈值：shadow model 作为伪 target，同一攻击函数在
  shadow_train(member)/shadow_test(nonmember) 上的分数经 §14.1 标准化校准
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Tuple

import numpy as np
import torch

from . import config as C
from . import data as D
from . import models as MO
from .bundles import load_reference_bundle, load_shadow_bundle
from .caching import load_cached_outputs
from .eval_common import evaluate_attack
from .training import per_sample_ce_loss

# 仓库攻击实现（复用，保证与仓库口径一致）
from Attack.base import AttackInput
from Attack.metric_based import _make_dummy_bench
from Attack.qmia import QMIAAttack
from Attack.rapid import RAPIDAttack
from Attack.shadow_based import ShadowBasedAttack
from Attack.utils_lira.lira_reference_utils import extract_true_label_confidences, logit_transform

ATTACK_FAMILY = {
    "LossAttack": "Metric", "CorrectnessAttack": "Metric", "ConfidenceAttack": "Metric",
    "EntropyAttack": "Metric", "ModifiedEntropyAttack": "Metric",
    "ShadowBasedAttack": "Classifier", "LiRAAttack": "Reference", "RMIAAttack": "Reference",
    "QMIAAttack": "Quantile", "RAPIDAttack": "Reference",
}


def _entr(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-30, None)
    return -(p * np.log(p)).sum(axis=1)


def _m_entr(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Song et al. modified entropy: (1 - p_y) * log2(sum_{z≠y} 2^{-p_z})
    # 与仓库 _m_entr_comp 同式
    n = probs.shape[0]
    p_y = probs[np.arange(n), y]
    modified = np.power(2, -probs)
    modified[np.arange(n), y] = 0.0
    return (1.0 - p_y) * np.log2(modified.sum(axis=1) + 1e-30)


def metric_family_scores(probs: np.ndarray, labels: np.ndarray, loss: np.ndarray) -> Dict[str, np.ndarray]:
    n = len(labels)
    return {
        "LossAttack": -np.asarray(loss, dtype=np.float64),
        "CorrectnessAttack": (probs.argmax(axis=1) == labels).astype(np.float64),
        "ConfidenceAttack": probs[np.arange(n), labels].astype(np.float64),
        "EntropyAttack": -_entr(probs),
        "ModifiedEntropyAttack": -_m_entr(probs, labels),
    }


def _ref_true_class_confs(ref_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ref_probs [R, N, C] → 真类置信度 [N, R]。"""
    R, N, _ = ref_probs.shape
    out = np.empty((N, R), dtype=np.float64)
    for r in range(R):
        out[:, r] = ref_probs[r][np.arange(N), labels]
    return out


def lira_offline_scores(target_probs: np.ndarray, labels: np.ndarray,
                        out_confs: np.ndarray) -> np.ndarray:
    """LiRA offline 变体（论文单侧 OUT 高斯）：score = (x − μ_out)/σ_out，logit 空间。

    out_confs: [N, R]（OUT reference 的真类置信度，NaN 表示该 ref 是 IN / 无效）。
    """
    conf = extract_true_label_confidences(np.asarray(target_probs, dtype=np.float64), labels)
    x = logit_transform(conf)
    outs = logit_transform(np.asarray(out_confs, dtype=np.float64))
    mu = np.nanmean(outs, axis=1)
    var = np.nanmean((outs - mu[:, None]) ** 2, axis=1)
    sd = np.sqrt(var)
    # σ 可能为 0（OUT ref 的置信度被同样裁剪到边界），z 会爆炸到 1e10 级，
    # 拖垮 shadow 均值校准；logit-z 超过 ±5 已是极端 member 证据，
    # 裁剪不改变评估侧排序（AUROC 不变），仅稳定阈值校准
    z = (x - mu) / np.maximum(sd, 1e-10)
    return np.clip(z, -5.0, 5.0)


def rmia_scores(target_eval_probs, y_eval, target_pop_probs, y_pop,
                ref_eval_confs, ref_pop_confs, a: float, gamma: float,
                chunk: int = 1024) -> np.ndarray:
    """RMIA（对齐 utils_rmia.compute_rmia_scores 公式，纯 offline：ref 全 OUT）。

    pr = 0.5*((1+a)*mean_ref_conf + (1-a))；score = mean_z[ ratio_x/ratio_z > gamma ]
    """
    conf_x = np.asarray(target_eval_probs, dtype=np.float64)[np.arange(len(y_eval)), y_eval]
    pr_x = 0.5 * ((1.0 + a) * np.mean(ref_eval_confs, axis=1) + (1.0 - a))
    ratio_x = conf_x / (pr_x + 1e-10)

    conf_z = np.asarray(target_pop_probs, dtype=np.float64)[np.arange(len(y_pop)), y_pop]
    pr_z = 0.5 * ((1.0 + a) * np.mean(ref_pop_confs, axis=1) + (1.0 - a))
    ratio_z = conf_z / (pr_z + 1e-10)

    scores = np.empty(len(y_eval), dtype=np.float64)
    for s in range(0, len(y_eval), chunk):
        block = ratio_x[s:s + chunk]
        counts = (block[:, None] / (ratio_z[None, :] + 1e-10) > gamma).sum(axis=1)
        scores[s:s + chunk] = counts / len(ratio_z)
    return scores


def _ce_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-30, 1.0)
    return -np.log(p[np.arange(len(labels)), labels])


class AttackSession:
    def __init__(self, dataset: str, seed: int, defense: str = "clean"):
        self.dataset, self.seed, self.defense = dataset, seed, defense
        self.cfg = C.load_config(dataset)
        self.manifest = C.load_manifest(dataset, seed)
        self.data = D.load_dataset(dataset)
        self.parts = D.partitions_from_manifest(self.data, self.manifest)
        self.is_image = C.DATASET_META[dataset]["is_image"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.target = load_cached_outputs(dataset, seed, defense)
        self.shadow = load_shadow_bundle(dataset, seed)
        self.ref = load_reference_bundle(dataset, seed)

        # 评估集视图
        me, nm = self.target["target_member_eval"], self.target["target_nonmember"]
        self.eval_probs = np.concatenate([me["probs"], nm["probs"]], axis=0)
        self.eval_labels = np.concatenate([me["labels"], nm["labels"]])
        self.eval_loss = np.concatenate([me["loss"], nm["loss"]])
        self.eval_membership = np.concatenate([np.ones(len(me["labels"])), np.zeros(len(nm["labels"]))])

        # ref 视图
        self.ref_eval_confs = np.concatenate([
            _ref_true_class_confs(self.ref["ref_probs_member_eval"].astype(np.float64), me["labels"]),
            _ref_true_class_confs(self.ref["ref_probs_nonmember"].astype(np.float64), nm["labels"]),
        ], axis=0)  # [N_eval, 4]

        # pool 位置映射（shadow 分区 → pool 行）
        pool_gids = self.ref["pool_indices"]
        self.pool_pos = {name: np.searchsorted(pool_gids, np.asarray(p["indices"], dtype=np.int64))
                         for name, p in self.manifest["partitions"].items()}

        # shadow 视图（member=shadow_train, nonmember=shadow_test）
        self.sh_mem_probs = self.shadow["shadow_train_probs"]
        self.sh_mem_y = self.shadow["shadow_train_labels"]
        self.sh_nm_probs = self.shadow["shadow_test_probs"]
        self.sh_nm_y = self.shadow["shadow_test_labels"]

        # RMIA population
        pop_pos = self.ref["population_pool_positions"]
        self.pop_probs = self.target["rmia_population"]["probs"]
        self.pop_y = self.target["rmia_population"]["labels"]
        self.ref_pop_confs = _ref_true_class_confs(
            self.ref["ref_probs_pool"][ :, pop_pos, :].astype(np.float64), self.pop_y)

    # ---------- LiRA offline 的 OUT conf 视图 ----------
    def _out_confs_for(self, pool_pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """pool 内若干样本的真类置信度 [N, 4]，IN ref 置 NaN。"""
        in_out = self.ref["in_out_matrix"]  # [4, P]
        sel = in_out[:, pool_pos]  # [4, N]
        confs = np.empty((len(pool_pos), in_out.shape[0]), dtype=np.float64)
        for r in range(in_out.shape[0]):
            rp = self.ref["ref_probs_pool"][r][pool_pos]  # [N, C]
            c = rp[np.arange(len(pool_pos)), labels].astype(np.float64)
            c[sel[r] == 1] = np.nan  # IN → 排除
            confs[:, r] = c
        return confs

    def _load_model(self, kind: str, name: str):
        cfg, meta = self.cfg, C.DATASET_META[self.dataset]
        input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))
        model = MO.build_model(cfg["model"]["arch"], input_dim=input_dim, num_classes=meta["num_classes"])
        path = C.CHECKPOINT_DIR / kind / self.dataset / f"{name}_seed{self.seed}.pt"
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()

    # ---------- 单攻击实现：返回 (eval_scores, shadow_member_scores, shadow_nonmember_scores) ----------
    def run_attack(self, attack: str, attack_seed: int):
        t0 = time.time()
        if attack in ("LossAttack", "CorrectnessAttack", "ConfidenceAttack", "EntropyAttack", "ModifiedEntropyAttack"):
            ev = metric_family_scores(self.eval_probs, self.eval_labels, self.eval_loss)
            sm = metric_family_scores(self.sh_mem_probs, self.sh_mem_y, self.shadow["shadow_train_loss"])
            sn = metric_family_scores(self.sh_nm_probs, self.sh_nm_y, self.shadow["shadow_test_loss"])
            return ev[attack], sm[attack], sn[attack]

        if attack == "LiRAAttack":
            ev = lira_offline_scores(self.eval_probs, self.eval_labels, self.ref_eval_confs)
            sm = lira_offline_scores(
                self.sh_mem_probs, self.sh_mem_y,
                self._out_confs_for(self.pool_pos["shadow_train"], self.sh_mem_y))
            sn = lira_offline_scores(
                self.sh_nm_probs, self.sh_nm_y,
                self._out_confs_for(self.pool_pos["shadow_test"], self.sh_nm_y))
            return ev, sm, sn

        if attack == "RMIAAttack":
            a = float(self.cfg["rmia"]["offline_a"]); gamma = float(self.cfg["rmia"]["gamma"])
            ev = rmia_scores(self.eval_probs, self.eval_labels, self.pop_probs, self.pop_y,
                             self.ref_eval_confs, self.ref_pop_confs, a, gamma)
            # shadow 侧：shadow model 对 population 的输出作为"target"信号
            sh_pop_probs = self.shadow["rmia_population_probs"]
            sh_pop_y = self.shadow["rmia_population_labels"]
            sm = rmia_scores(self.sh_mem_probs, self.sh_mem_y, sh_pop_probs, sh_pop_y,
                             self._all_confs_for(self.pool_pos["shadow_train"], self.sh_mem_y),
                             self.ref_pop_confs, a, gamma)
            sn = rmia_scores(self.sh_nm_probs, self.sh_nm_y, sh_pop_probs, sh_pop_y,
                             self._all_confs_for(self.pool_pos["shadow_test"], self.sh_nm_y),
                             self.ref_pop_confs, a, gamma)
            return ev, sm, sn

        if attack == "QMIAAttack":
            torch.manual_seed(attack_seed)
            model = self._load_model("targets", "target") if self.defense == "clean" else self._load_defended()
            aux_X, aux_y = self.parts["auxiliary"]
            ev = self._qmia(model, aux_X, aux_y,
                            np.concatenate([self.parts["target_member_eval"][0], self.parts["target_nonmember"][0]]),
                            self.eval_labels)
            shadow_model = self._load_model("shadows", "shadow")
            sm = self._qmia(shadow_model, aux_X, aux_y, self.parts["shadow_train"][0], self.sh_mem_y)
            sn = self._qmia(shadow_model, aux_X, aux_y, self.parts["shadow_test"][0], self.sh_nm_y)
            return ev, sm, sn

        if attack == "ShadowBasedAttack":
            torch.manual_seed(attack_seed)
            atk = ShadowBasedAttack(device=self.device)
            atk.fit(AttackInput(None, None, shadow_data={
                "member_outputs": self.sh_mem_probs, "member_labels": self.sh_mem_y,
                "nonmember_outputs": self.sh_nm_probs, "nonmember_labels": self.sh_nm_y,
            }))
            ev = atk.infer(AttackInput(None, None, signals={"probabilities": self.eval_probs}, labels=self.eval_labels)).membership_scores
            sh_all = np.concatenate([self.sh_mem_probs, self.sh_nm_probs], axis=0)
            sh_labels = np.concatenate([self.sh_mem_y, self.sh_nm_y])
            sh_side = atk.infer(AttackInput(None, None, signals={"probabilities": sh_all}, labels=sh_labels)).membership_scores
            m, n = len(self.sh_mem_y), len(self.sh_nm_y)
            return np.asarray(ev, dtype=np.float64), np.asarray(sh_side[:m], dtype=np.float64), np.asarray(sh_side[m:], dtype=np.float64)

        if attack == "RAPIDAttack":
            torch.manual_seed(attack_seed)
            # orig = -CE(model)；ref_mean = 4 refs 的 -CE 均值；calibrated = orig − ref_mean
            ref_eval = np.concatenate(
                [self.ref["ref_probs_member_eval"], self.ref["ref_probs_nonmember"]], axis=1)  # [R,N,C]
            ev_orig = -self.eval_loss
            ev_cal = ev_orig - _ref_mean_negce(ref_eval, self.eval_labels)

            mem_orig = -np.asarray(self.shadow["shadow_train_loss"], dtype=np.float64)
            mem_cal = mem_orig - _ref_mean_negce(
                self.ref["ref_probs_pool"][:, self.pool_pos["shadow_train"], :], self.sh_mem_y)
            nm_orig = -np.asarray(self.shadow["shadow_test_loss"], dtype=np.float64)
            nm_cal = nm_orig - _ref_mean_negce(
                self.ref["ref_probs_pool"][:, self.pool_pos["shadow_test"], :], self.sh_nm_y)

            # RAPID 的 fit 无条件要求 reference_data；signals 路径不使用 manager，
            # 传一个由 Bundle 数据构建的未训练 stub（协议：ref 信号全部来自 Reference Bundle）
            from Attack.utils_rapid.rapid_reference_utils import RAPIDReferenceManager
            pool_X, pool_y = self.parts["reference_pool"]
            stub = RAPIDReferenceManager(
                train_X=pool_X, train_y=pool_y,
                test_X=self.parts["target_member_eval"][0], test_y=self.parts["target_member_eval"][1],
                model_factory=lambda: None, device=str(self.device),
            )
            atk = RAPIDAttack(device=self.device)
            atk.fit(AttackInput(None, None, shadow_data={
                "original_scores": np.concatenate([mem_orig, nm_orig]),
                "calibrated_scores": np.concatenate([mem_cal, nm_cal]),
                "membership_labels": np.concatenate([np.ones(len(mem_orig)), np.zeros(len(nm_orig))]),
            }, reference_data={"reference_manager": stub}))
            ev = np.asarray(atk.infer(AttackInput(None, None, signals={
                "original_scores": ev_orig, "calibrated_scores": ev_cal}, labels=self.eval_labels)).membership_scores, dtype=np.float64)
            sh_labels = np.concatenate([self.sh_mem_y, self.sh_nm_y])
            sh_side = np.asarray(atk.infer(AttackInput(None, None, signals={
                "original_scores": np.concatenate([mem_orig, nm_orig]),
                "calibrated_scores": np.concatenate([mem_cal, nm_cal])}, labels=sh_labels)).membership_scores, dtype=np.float64)
            m, n = len(mem_orig), len(nm_orig)
            return ev, sh_side[:m], sh_side[m:]

        raise ValueError(f"unknown attack: {attack}")

    # ---------- helpers ----------
    def _all_confs_for(self, pool_pos: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """[N, 4] 全部 ref 的真类置信度（RMIA 的 pr 估计用 in∪out，与仓库一致）。"""
        confs = np.empty((len(pool_pos), self.ref["in_out_matrix"].shape[0]), dtype=np.float64)
        for r in range(confs.shape[1]):
            rp = self.ref["ref_probs_pool"][r][pool_pos]
            confs[:, r] = rp[np.arange(len(pool_pos)), labels].astype(np.float64)
        return confs

    def _qmia(self, model, fit_X, fit_y, X, y):
        atk = QMIAAttack(device=self.device)
        atk.fit(AttackInput(model, None, shadow_data={"fit_X": fit_X, "fit_y": fit_y}))
        out = atk.infer(AttackInput(model, X, labels=y))
        return np.asarray(out.membership_scores, dtype=np.float64)

    def _load_defended(self):
        path = C.CHECKPOINT_DIR / "defenses" / self.dataset / f"{self.defense}_seed{self.seed}.pt"
        if self.defense in ("MemGuard", "HAMP"):
            # MemGuard / HAMP 的 protected predictor 是模型包装器，整对象 pickle
            obj = torch.load(path, map_location=self.device, weights_only=False)
            return obj.to(self.device).eval() if hasattr(obj, "to") else obj
        cfg, meta = self.cfg, C.DATASET_META[self.dataset]
        input_dim = meta.get("input_dim", 0) or int(cfg["data"].get("input_dim", 0))
        model = MO.build_model(cfg["model"]["arch"], input_dim=input_dim, num_classes=meta["num_classes"])
        model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return model.to(self.device).eval()


def _ref_mean_negce(ref_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """ref_probs [R, N, C] → 各 ref 的 -CE 均值 [N]。"""
    R, N, _ = ref_probs.shape
    total = np.zeros(N, dtype=np.float64)
    for r in range(R):
        p = np.clip(ref_probs[r].astype(np.float64), 1e-30, 1.0)
        total += -np.log(p[np.arange(N), labels])
    return total / R


def run_all_attacks(dataset: str, seed: int, defense: str = "clean",
                    attacks=None, stage: str = "clean_attack") -> list:
    """运行全部攻击并落盘 results/<stage>/<dataset>/seed<k>/attacks.json。"""
    session = AttackSession(dataset, seed, defense)
    attack_seed = 100 * seed + 90  # 攻击头训练种子（派生规则记录在结果里）
    rows = []
    for attack in (attacks or session.cfg["attack"]["attacks"]):
        t0 = time.time()
        ev, sm, sn = session.run_attack(attack, attack_seed)
        metrics = evaluate_attack(ev, session.eval_membership, sm, sn, session.is_image)
        row = {
            "dataset": dataset, "seed": seed, "defense": defense, "attack": attack,
            "family": ATTACK_FAMILY[attack], "attack_seed": attack_seed,
            **metrics,
            "attack_seconds": round(time.time() - t0, 2),
            "git_commit": session.manifest["generated"]["git_commit"],
            "params": {
                "rmia": session.cfg.get("rmia", {}),
                "lira_variant": "offline_out_gaussian_zscore",
            },
        }
        rows.append(row)
        print(f"[attack] {dataset}/s{seed}/{defense}/{attack}: "
              f"AUROC={metrics['auroc']:.4f} Acc={metrics['accuracy']:.4f} "
              f"TPR@1%={metrics['tpr_at_1pct_fpr']:.4f} ({row['attack_seconds']}s)")

    out_dir = C.RESULTS_DIR / stage / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"attacks_{defense}.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=1)
    return rows
