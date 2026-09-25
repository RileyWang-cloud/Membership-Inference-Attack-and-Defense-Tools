"""Attack execution layer (plan §8, STEP 6 / STEP 10).

All ten attacks reuse the repository's own implementations (user directive:
run with the repo's code). Attack-side resources come exclusively from the
shared Shadow / Reference bundles (plan §9, §13 non-adaptive threat model):

- A01-A05 metric attacks       -> repo ``Attack/metric_based.py`` (signals
  from the Stage-I output cache; Song-format shadow calibration).
- A06 ShadowBasedAttack        -> repo ``Attack/shadow_based.py`` format B.
- A07 LiRA (offline)           -> repo ``Attack/utils_lira`` manager hydrated
  from the reference bundle; offline LLR = -log p_out in logit space (§8.4).
- A08 RMIA                     -> repo ``Attack/rmia.py`` with hydrated
  manager, gamma=2, offline_a=0.3 (§4.4).
- A09 QMIA                     -> repo ``Attack/qmia.py`` (image-input fix W6).
- A10 RAPID                    -> repo ``Attack/rapid.py`` with precomputed
  shadow features + difficulty calibration from the reference bundle.

Shadow-side calibration scores (mu_m / mu_nm for the §14.1 accuracy
threshold) are computed once per seed by running the same scoring path with
the shadow model as "target", and cached under bundles/seed<k>/calibration/.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from Attack.eval_common import evaluate_scores
from benchmark.bmk import common
from benchmark.bmk.data import gather_global, load_cifar10_tensors
from benchmark.bmk.models import resnet18_cifar_factory, resnet18_cifar_groupnorm_factory
from benchmark.bmk.train import load_cache

RMIA_GAMMA = 2.0
RMIA_OFFLINE_A = 0.3  # CIFAR-10 value from the RMIA paper (plan §4.4)

ATTACK_ORDER = [
    "LossAttack", "CorrectnessAttack", "ConfidenceAttack", "EntropyAttack",
    "ModifiedEntropyAttack", "ShadowBasedAttack", "LiRAAttack", "RMIAAttack",
    "QMIAAttack", "RAPIDAttack",
]
REATTACK_ORDER = ["LossAttack", "ShadowBasedAttack", "LiRAAttack", "RMIAAttack", "QMIAAttack"]
ATTACK_FAMILY = {
    "LossAttack": "metric", "CorrectnessAttack": "metric", "ConfidenceAttack": "metric",
    "EntropyAttack": "metric", "ModifiedEntropyAttack": "metric",
    "ShadowBasedAttack": "classifier", "LiRAAttack": "reference",
    "RMIAAttack": "reference", "QMIAAttack": "learned_quantile", "RAPIDAttack": "reference_augmentation",
}


# ----------------------------------------------------------------- context
class AttackContext:
    """Everything a scoring pass needs for one (seed, defense) pair."""

    def __init__(self, seed: int, defense: str = "clean", load_raw: bool = False) -> None:
        self.seed = seed
        self.defense = defense
        self.manifest = common.load_manifest(seed)
        self.shadow_npz = np.load(common.bundle_dir(seed) / "shadow_bundle.npz")
        self.ref_npz = np.load(common.bundle_dir(seed) / "reference_bundle.npz")
        self.member_eval = load_cache(seed, defense, "target_member_eval")
        self.nonmember = load_cache(seed, defense, "target_nonmember")
        self.labels = np.concatenate([self.member_eval["labels"], self.nonmember["labels"]])
        self.membership = np.concatenate(
            [np.ones(len(self.member_eval["labels"])), np.zeros(len(self.nonmember["labels"]))]
        ).astype(np.int64)
        self.signals = {
            "logits": np.concatenate([self.member_eval["logits"], self.nonmember["logits"]]),
            "probabilities": np.concatenate([self.member_eval["probs"], self.nonmember["probs"]]),
        }
        self._tensors = load_cifar10_tensors() if load_raw else None
        self._managers: Dict[str, object] = {}

    @property
    def tensors(self):
        if self._tensors is None:
            self._tensors = load_cifar10_tensors()
        return self._tensors

    def images_for(self, partition: str) -> Tuple[torch.Tensor, torch.Tensor]:
        return gather_global(self.manifest["partitions"][partition]["global_indices"], self.tensors)

    # -- manager hydration (plan §9.2: hydrate, never retrain) -------------
    def pool_positions_of(self, partition: str) -> np.ndarray:
        """Pool positions occupied by a train-side partition (shadow_train etc.)."""
        pool_global = self.ref_npz["pool_to_global"]
        wanted = set(self.manifest["partitions"][partition]["global_indices"])
        lookup = {int(g): p for p, g in enumerate(pool_global)}
        return np.sort(np.asarray([lookup[g] for g in wanted]))

    def _hydrate_reference_predictions(self) -> Dict[int, dict]:
        """Populate {pool_pos: {"in": [...], "out": [...]}} + eval entries."""
        ref_pool = self.ref_npz["ref_pool_probabilities"]  # [4, N_pool, C]
        ref_eval = self.ref_npz["ref_probabilities"]       # [4, N_eval, C]
        in_out = self.ref_npz["ref_in_out_matrix"]         # [4, N_pool]
        n_pool, n_eval = ref_pool.shape[1], ref_eval.shape[1]
        preds = {"in": {}, "out": {}}
        for r in range(common.NUM_REFS):
            member_mask = in_out[r]
            for p in np.where(member_mask)[0]:
                preds["in"].setdefault(int(p), []).append(ref_pool[r, p].astype(np.float64))
            for p in np.where(~member_mask)[0]:
                preds["out"].setdefault(int(p), []).append(ref_pool[r, p].astype(np.float64))
            for e in range(n_eval):  # eval samples are OUT for every ref (offline protocol)
                preds["out"].setdefault(n_pool + e, []).append(ref_eval[r, e].astype(np.float64))
        return preds

    def hydrated_manager(self, kind: str):
        """Build a pre-hydrated LiRA/RMIA reference manager (index space:
        pool positions [0, 25k), eval rows [25k, 45k))."""
        if kind in self._managers:
            return self._managers[kind]
        if kind == "rmia":
            from Attack.utils_rmia.rmia_reference_utils import RMIAReferenceManager as M
        elif kind == "lira":
            from Attack.utils_lira.lira_reference_utils import LiRAReferenceManager as M
        else:
            raise ValueError(kind)
        ref = self.ref_npz
        n_pool, n_eval = ref["ref_pool_probabilities"].shape[1], ref["ref_probabilities"].shape[1]
        manager = M(
            train_X=np.zeros((n_pool, 1), dtype=np.float32),
            train_y=np.asarray(ref["pool_labels"]),
            test_X=np.zeros((n_eval, 1), dtype=np.float32),
            test_y=np.asarray(ref["eval_labels"]),
            model_factory=lambda: None,
        )
        preds = self._hydrate_reference_predictions()
        manager.reference_predictions.in_model_predictions = preds["in"]
        manager.reference_predictions.out_model_predictions = preds["out"]
        manager.reference_predictions.sample_labels = np.concatenate(
            [np.asarray(ref["pool_labels"]), np.asarray(ref["eval_labels"])]
        ).astype(np.int64)
        self._managers[kind] = manager
        return manager

    def shadow_song_data(self) -> Dict[str, np.ndarray]:
        """Shadow bundle in the Song et al. bench format used by the repo's
        metric attacks (plan §8.2 shadow calibration)."""
        return {
            "s_tr_outputs": self.shadow_npz["shadow_member_outputs"],
            "s_tr_labels": self.shadow_npz["shadow_member_labels"],
            "s_te_outputs": self.shadow_npz["shadow_nonmember_outputs"],
            "s_te_labels": self.shadow_npz["shadow_nonmember_labels"],
            "t_tr_outputs": self.member_eval["probs"],
            "t_tr_labels": self.member_eval["labels"],
            "t_te_outputs": self.nonmember["probs"],
            "t_te_labels": self.nonmember["labels"],
            "num_classes": common.NUM_CLASSES,
        }


# ------------------------------------------------------------ LiRA offline
def lira_offline_scores(manager, target_probs: np.ndarray, sample_indices: np.ndarray,
                        true_labels: np.ndarray) -> np.ndarray:
    """Offline LiRA LLR in logit space (plan §8.4): score = -log p_out.

    Uses the repo manager's own get_sample_predictions + logit_transform;
    with the offline protocol every eval sample has only OUT references, so
    the Gaussian is fitted on the OUT confidences alone (Carlini et al. 2021,
    offline variant).
    """
    from scipy.stats import norm

    from Attack.utils_lira.lira_reference_utils import logit_transform

    ref_data = manager.get_sample_predictions(np.asarray(sample_indices, dtype=np.int64))
    out_confs = ref_data["out_confs"]  # (n, n_out_refs), NaN-padded
    if out_confs.shape[1] == 0:
        raise RuntimeError("offline LiRA requires at least one OUT reference per sample")
    out_logit = logit_transform(out_confs)
    counts = np.sum(~np.isnan(out_logit), axis=1)
    mean = np.nanmean(out_logit, axis=1)
    var = np.nanvar(out_logit, axis=1)
    var[counts < 2] = np.nan  # a single observation has no variance; fall back below
    target_confs = target_probs[np.arange(len(target_probs)), np.asarray(true_labels, dtype=np.int64)]
    target_logit = logit_transform(target_confs)
    sigma = np.sqrt(var)
    valid = ~np.isnan(mean)
    scores = np.zeros(len(sample_indices), dtype=np.float64)
    # single-observation rows: sigma = 0 -> logpdf degenerates; use global median sigma
    fallback_sigma = float(np.nanmedian(sigma)) if np.any(~np.isnan(sigma)) else 0.1
    mu = np.where(np.isnan(mean), np.nanmean(mean), mean)
    sig = np.where(np.isnan(sigma), max(fallback_sigma, 1e-6), np.maximum(sigma, 1e-10))
    scores[valid] = -norm.logpdf(target_logit[valid], loc=mu[valid], scale=sig[valid])
    return scores


# ---------------------------------------------------------------- scoring
def _metric_attack(ctx: AttackContext, name: str):
    """Run one repo metric attack (A01-A05) on the cached target outputs."""
    import Attack.metric_based as mb

    common.set_global_seed(9000 + ctx.seed)
    cls = {
        "LossAttack": mb.LossAttack, "CorrectnessAttack": mb.CorrectnessAttack,
        "ConfidenceAttack": mb.ConfidenceAttack, "EntropyAttack": mb.EntropyAttack,
        "ModifiedEntropyAttack": mb.ModifiedEntropyAttack,
    }[name]
    attack = cls(batch_size=512)
    from Attack.metric_based import AttackInput  # repo-local input type

    needs_shadow = name in {"ConfidenceAttack", "EntropyAttack", "ModifiedEntropyAttack"}
    attack_input = AttackInput(
        target_model=None,
        samples=None,
        labels=ctx.labels,
        signals=ctx.signals,
        shadow_data=ctx.shadow_song_data() if needs_shadow else None,
    )
    attack.fit(attack_input)
    output = attack.infer(attack_input)
    return output.membership_scores, attack, attack_input


def _metric_shadow_side(ctx: AttackContext, name: str, attack, attack_input):
    """Shadow-side scores for threshold calibration: same fitted attack,
    signals replaced by the shadow model's own outputs (plan §14.1)."""
    from Attack.metric_based import AttackInput

    shadow = ctx.shadow_npz
    def scores_for(outputs, labels):
        inp = AttackInput(
            target_model=None, samples=None, labels=labels,
            signals={"probabilities": outputs},
        )
        out = attack.infer(inp)
        return np.asarray(out.membership_scores, dtype=np.float64)

    mu_m = scores_for(shadow["shadow_member_outputs"], shadow["shadow_member_labels"])
    mu_nm = scores_for(shadow["shadow_nonmember_outputs"], shadow["shadow_nonmember_labels"])
    return mu_m, mu_nm


def score_attack(ctx: AttackContext, name: str):
    """Return (eval_scores, shadow_member_scores, shadow_nonmember_scores)."""
    if name in {"LossAttack", "CorrectnessAttack", "ConfidenceAttack", "EntropyAttack",
                "ModifiedEntropyAttack"}:
        scores, attack, attack_input = _metric_attack(ctx, name)
        scores = np.asarray(scores, dtype=np.float64)
        if name == "LossAttack":
            mu_m = -ctx.shadow_npz["shadow_member_losses"].astype(np.float64)
            mu_nm = -ctx.shadow_npz["shadow_nonmember_losses"].astype(np.float64)
        elif name == "CorrectnessAttack":
            s = ctx.shadow_npz
            mu_m = (s["shadow_member_outputs"].argmax(1) == s["shadow_member_labels"]).astype(np.float64)
            mu_nm = (s["shadow_nonmember_outputs"].argmax(1) == s["shadow_nonmember_labels"]).astype(np.float64)
        else:
            mu_m, mu_nm = _metric_shadow_side(ctx, name, attack, attack_input)
        return scores, mu_m, mu_nm

    if name == "ShadowBasedAttack":
        import Attack.shadow_based as sb

        common.set_global_seed(9100 + ctx.seed)
        attack = sb.ShadowBasedAttack(batch_size=512)
        from Attack.shadow_based import AttackInput

        shadow = ctx.shadow_npz
        attack_input = AttackInput(
            target_model=None, samples=None, labels=ctx.labels, signals=ctx.signals,
            shadow_data={
                "member_outputs": shadow["shadow_member_outputs"],
                "member_labels": shadow["shadow_member_labels"],
                "nonmember_outputs": shadow["shadow_nonmember_outputs"],
                "nonmember_labels": shadow["shadow_nonmember_labels"],
            },
            config={"num_classes": common.NUM_CLASSES},
        )
        attack.fit(attack_input)
        out = attack.infer(attack_input)

        def shadow_side(probs, labels):
            inp = AttackInput(target_model=None, samples=None, labels=labels,
                              signals={"probabilities": probs})
            return np.asarray(attack.infer(inp).membership_scores, dtype=np.float64)

        mu_m = shadow_side(shadow["shadow_member_outputs"], shadow["shadow_member_labels"])
        mu_nm = shadow_side(shadow["shadow_nonmember_outputs"], shadow["shadow_nonmember_labels"])
        return np.asarray(out.membership_scores, dtype=np.float64), mu_m, mu_nm

    if name == "LiRAAttack":
        manager = ctx.hydrated_manager("lira")
        sample_indices = np.arange(
            ctx.ref_npz["ref_pool_probabilities"].shape[1],
            ctx.ref_npz["ref_pool_probabilities"].shape[1] + len(ctx.labels),
        )
        scores = lira_offline_scores(manager, ctx.signals["probabilities"], sample_indices, ctx.labels)
        # shadow side: shadow model as target over its own member/non-member sets
        shadow = ctx.shadow_npz
        pos_m = ctx.pool_positions_of("shadow_train")
        pos_nm = ctx.pool_positions_of("shadow_test")
        mu_m = lira_offline_scores(manager, shadow["shadow_member_outputs"], pos_m,
                                   shadow["shadow_member_labels"])
        mu_nm = lira_offline_scores(manager, shadow["shadow_nonmember_outputs"], pos_nm,
                                    shadow["shadow_nonmember_labels"])
        return scores, np.nan_to_num(mu_m), np.nan_to_num(mu_nm)

    if name == "RMIAAttack":
        import Attack.rmia as rm

        common.set_global_seed(9200 + ctx.seed)
        attack = rm.RMIAAttack(gamma=RMIA_GAMMA, a=RMIA_OFFLINE_A, batch_size=512)
        from Attack.rmia import AttackInput

        n_pool = ctx.ref_npz["ref_pool_probabilities"].shape[1]
        # target-side population probabilities come from the Stage-I output cache;
        # reference-side Pr(z) is handled inside the hydrated manager
        pop_cache = load_cache(ctx.seed, ctx.defense, "rmia_population")
        attack_input = AttackInput(
            target_model=None, samples=None, labels=ctx.labels, signals={
                "probabilities": ctx.signals["probabilities"],
                "population_probabilities": pop_cache["probs"],
            },
            reference_data={"reference_manager": ctx.hydrated_manager("rmia"),
                            "population_labels": pop_cache["labels"]},
            metadata={
                "sample_indices": np.arange(n_pool, n_pool + len(ctx.labels)),
                "population_indices": np.asarray(ctx.manifest["rmia_population"]["pool_positions"]),
            },
            config={"a": RMIA_OFFLINE_A, "gamma": RMIA_GAMMA},
        )
        out = attack.run(attack_input)
        # shadow side (shadow model as target over its own train/test; its own
        # outputs on the population z are stored in the shadow bundle)
        shadow = ctx.shadow_npz
        def rmia_side(probs, pool_pos, labels):
            inp = AttackInput(
                target_model=None, samples=None, labels=labels,
                signals={"probabilities": probs,
                         "population_probabilities": shadow["shadow_population_outputs"]},
                reference_data={"reference_manager": ctx.hydrated_manager("rmia"),
                                "population_labels": shadow["shadow_population_labels"]},
                metadata={"sample_indices": pool_pos,
                          "population_indices": np.asarray(ctx.manifest["rmia_population"]["pool_positions"])},
                config={"a": RMIA_OFFLINE_A, "gamma": RMIA_GAMMA},
            )
            return np.asarray(attack.infer(inp).membership_scores, dtype=np.float64)

        mu_m = rmia_side(shadow["shadow_member_outputs"], ctx.pool_positions_of("shadow_train"),
                        shadow["shadow_member_labels"])
        mu_nm = rmia_side(shadow["shadow_nonmember_outputs"], ctx.pool_positions_of("shadow_test"),
                          shadow["shadow_nonmember_labels"])
        return np.asarray(out.membership_scores, dtype=np.float64), mu_m, mu_nm

    if name == "QMIAAttack":
        from Attack.qmia import QMIAAttack

        device = common.resolve_device()
        model = load_target_module(ctx)
        common.set_global_seed(9300 + ctx.seed)
        attack = QMIAAttack(n_epochs=30, batch_size=256)
        fit_X, fit_y = ctx.images_for("auxiliary")
        from Attack.base import AttackInput

        X_eval = torch.cat([ctx.images_for("target_member_eval")[0], ctx.images_for("target_nonmember")[0]])
        attack_input = AttackInput(
            target_model=model, samples=X_eval, labels=ctx.labels,
            shadow_data={"fit_X": fit_X, "fit_y": fit_y},
            config={"batch_size": 512},
        )
        out = attack.run(attack_input)
        # shadow side: the SAME attack recipe applied with the shadow model as target
        shadow_model = resnet18_cifar_factory()
        sckpt = torch.load(common.checkpoint_dir("shadows") / f"shadow_seed{ctx.seed}.pt",
                           map_location="cpu", weights_only=False)
        shadow_model.load_state_dict(sckpt["state_dict"])
        common.set_global_seed(9301 + ctx.seed)
        shadow_attack = QMIAAttack(n_epochs=30, batch_size=256)
        X_m, y_m = ctx.images_for("shadow_train")
        X_nm, y_nm = ctx.images_for("shadow_test")
        out_m = shadow_attack.run(AttackInput(
            target_model=shadow_model, samples=X_m, labels=y_m,
            shadow_data={"fit_X": fit_X, "fit_y": fit_y}, config={"batch_size": 512},
        ))
        out_nm = shadow_attack.infer(AttackInput(
            target_model=shadow_model, samples=X_nm, labels=y_nm,
            shadow_data={"fit_X": fit_X, "fit_y": fit_y}, config={"batch_size": 512}))
        return (np.asarray(out.membership_scores, dtype=np.float64),
                np.asarray(out_m.membership_scores, dtype=np.float64),
                np.asarray(out_nm.membership_scores, dtype=np.float64))

    if name == "RAPIDAttack":
        import Attack.rapid as rp
        from Attack.rapid import AttackInput

        common.set_global_seed(9400 + ctx.seed)
        attack = rp.RAPIDAttack(batch_size=512)
        # difficulty calibration from the reference bundle (plan §8.4)
        ref_eval = ctx.ref_npz["ref_probabilities"]            # [4, N_eval, C]
        ref_pool = ctx.ref_npz["ref_pool_probabilities"]       # [4, N_pool, C]
        labels_eval = ctx.labels
        log_p_true_eval = _mean_log_true(ref_eval, labels_eval)

        original = -np.concatenate([ctx.member_eval["loss"], ctx.nonmember["loss"]]).astype(np.float64)
        calibrated = original - log_p_true_eval

        shadow = ctx.shadow_npz
        pos_m = ctx.pool_positions_of("shadow_train")
        pos_nm = ctx.pool_positions_of("shadow_test")
        orig_m = -shadow["shadow_member_losses"].astype(np.float64)
        orig_nm = -shadow["shadow_nonmember_losses"].astype(np.float64)
        feat_m = np.stack([orig_m, orig_m - _mean_log_true(ref_pool[:, pos_m], shadow["shadow_member_labels"])], axis=1)
        feat_nm = np.stack([orig_nm, orig_nm - _mean_log_true(ref_pool[:, pos_nm], shadow["shadow_nonmember_labels"])], axis=1)
        attack_input = AttackInput(
            target_model=None, samples=None, labels=labels_eval,
            signals={"original_scores": original, "calibrated_scores": calibrated},
            shadow_data={
                "original_scores": np.concatenate([feat_m[:, 0], feat_nm[:, 0]]),
                "calibrated_scores": np.concatenate([feat_m[:, 1], feat_nm[:, 1]]),
                "membership_labels": np.concatenate([np.ones(len(feat_m)), np.zeros(len(feat_nm))]),
            },
            reference_data={"reference_manager": _rapid_stub_manager(ctx)},
        )
        attack.fit(attack_input)
        out = attack.infer(attack_input)

        def side(features):
            inp = AttackInput(target_model=None, samples=None, labels=np.zeros(len(features), dtype=np.int64),
                              signals={"original_scores": features[:, 0], "calibrated_scores": features[:, 1]})
            return np.asarray(attack.infer(inp).membership_scores, dtype=np.float64)

        return (np.asarray(out.membership_scores, dtype=np.float64), side(feat_m), side(feat_nm))

    raise ValueError(f"unknown attack {name}")


def _mean_log_true(ref_probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Mean over reference models of log p_true (RAPID reference mean score)."""
    picked = ref_probs[:, np.arange(len(labels)), np.asarray(labels, dtype=np.int64)]
    return np.log(np.clip(picked, 1e-12, None)).mean(axis=0)


def _rapid_stub_manager(ctx: AttackContext):
    """RAPID manager whose scoring methods are never invoked: all features
    are precomputed from the reference bundle (plan §8.4)."""
    from Attack.utils_rapid.rapid_reference_utils import RAPIDReferenceManager

    ref = ctx.ref_npz
    return RAPIDReferenceManager(
        train_X=np.zeros((ref["ref_pool_probabilities"].shape[1], 1), dtype=np.float32),
        train_y=np.asarray(ref["pool_labels"]),
        test_X=np.zeros((ref["ref_probabilities"].shape[1], 1), dtype=np.float32),
        test_y=np.asarray(ref["eval_labels"]),
        model_factory=lambda: None,
    )


def _defense_checkpoint(ctx: AttackContext) -> Path:
    if ctx.defense == "clean":
        return common.checkpoint_dir("targets") / f"target_seed{ctx.seed}.pt"
    return common.checkpoint_dir("defenses") / f"{ctx.defense}_seed{ctx.seed}.pt"


def load_target_module(ctx: AttackContext):
    """Load the (possibly wrapped) target model for QMIA.

    Wrapper defenses (HAMP modify_output, MemGuard) persist the full wrapper
    module under ``full_module``; plain ResNet18 checkpoints are rebuilt from
    ``state_dict``.
    """
    ckpt = torch.load(_defense_checkpoint(ctx), map_location="cpu", weights_only=False)
    if ckpt.get("full_module") is not None:
        return ckpt["full_module"]
    # D01 trains the GroupNorm variant (plan §10.4) — its state_dict does not
    # fit the BatchNorm factory.
    factory = (resnet18_cifar_groupnorm_factory
               if ctx.defense == "D01_DP_SGD" else resnet18_cifar_factory)
    model = factory()
    model.load_state_dict(ckpt["state_dict"])
    return model


def run_one_attack(ctx: AttackContext, name: str, result_extra: dict) -> dict:
    scores, mu_m, mu_nm = score_attack(ctx, name)
    metrics = evaluate_scores(scores, ctx.membership, mu_m, mu_nm, image_dataset=True)
    record = {
        "dataset": common.DATASET,
        "model": "resnet18_cifar",
        "seed": ctx.seed,
        "defense": ctx.defense,
        "attack": name,
        "family": ATTACK_FAMILY[name],
        "rmia_config": {"gamma": RMIA_GAMMA, "offline_a": RMIA_OFFLINE_A},
        **metrics,
        **result_extra,
    }
    return record
