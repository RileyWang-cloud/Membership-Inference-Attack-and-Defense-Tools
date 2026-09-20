"""MemGuard output-perturbation defense for PyTorch classifiers.

Reference: Jia, Gong, Bi, Li and Cao, "MemGuard: Defending against Black-Box
Membership Inference Attacks via Adversarial Examples", CCS 2019.

MemGuard is an inference-time defense: it leaves the model untouched and
instead perturbs every posterior (probability vector) that the model releases.
The per-sample perturbation is chosen so that

1. utility is preserved -- the perturbed posterior stays on the probability
   simplex, keeps the original argmax prediction, and keeps the
   predicted-class probability above a confidence floor (the paper's
   ``0.5 + epsilon`` zero-loss condition), and
2. the membership signal is destroyed -- the entropy of the perturbed
   posterior is driven onto a calibration target estimated from non-member
   posteriors, so confidence / entropy / loss based attacks no longer
   separate members from non-members.  The defense cannot tell members from
   non-members at deployment time, so *every* released posterior is moved
   onto the same entropy target from either side, which is also the fixed
   point of the paper's Lagrangian alternation.

The paper solves this with a per-sample Lagrangian
``min_delta f(p + delta) - lambda * g(p + delta)`` and alternating updates of
``delta`` and ``lambda``.  This implementation notes that the fixed point has
a closed form: among posteriors with entropy ``T`` and predicted-class
probability at least the floor, the canonical solution -- and the limit of
the alternating optimization -- is ``[c, (1-c)/(K-1), ...]`` placed on the
original argmax, where ``c`` is the unique confidence with entropy ``T``.
``c`` is found by a scalar bisection (the same construction HAMP uses for its
high-entropy training targets), which makes the defense deterministic,
vectorized over the whole batch, and free of optimizer tuning.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from Defense._classification import (
    binary_auroc,
    classifier_metrics,
    extract_logits,
    predict_logits,
    resolve_device,
    to_label_tensor,
)
from Defense.base import BaseDefense, DefenseEvaluationResult, DefenseInput, DefenseOutput


def entropy_of_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Natural entropy of each row of a probability matrix.

    Sibling of ``_classification.probability_entropy`` (which softmaxes a
    logits matrix first); keep the two formulas in sync.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, classes).")
    clamped = probabilities.clamp_min(1e-12)
    return -(probabilities * clamped.log()).sum(dim=1)


def entropy_at_confidence(confidence: float, num_classes: int) -> float:
    """Entropy of ``[confidence, (1-confidence)/(K-1), ...]``."""
    if num_classes < 2:
        raise ValueError("MemGuard requires at least two classes.")
    confidence = float(np.clip(confidence, 1e-12, 1.0))
    other = (1.0 - confidence) / (num_classes - 1)
    entropy = -confidence * np.log(confidence)
    if other > 0.0:
        entropy -= (num_classes - 1) * other * np.log(other)
    return float(entropy)


def max_entropy_at_floor(num_classes: int, confidence_floor: float) -> float:
    """Highest entropy reachable while keeping the top probability at ``floor``.

    This is the entropy of ``[floor, (1-floor)/(K-1), ...]`` and upper-bounds
    the entropy target so that the perturbation always stays solvable.
    """
    if not 0.0 < confidence_floor < 1.0:
        raise ValueError("confidence_floor must lie strictly between 0 and 1.")
    return entropy_at_confidence(confidence_floor, num_classes)


def confidence_for_entropy(
    entropy_target: float,
    num_classes: int,
    confidence_floor: float,
    bisection_iterations: int = 80,
) -> float:
    """Confidence ``c`` such that ``[c, (1-c)/(K-1), ...]`` has the target entropy.

    ``entropy_at_confidence`` is strictly decreasing in ``c``, so a bisection
    recovers ``c`` exactly.  The returned confidence respects the floor and
    stays strictly above the uniform share, which keeps the argmax intact.
    """
    if num_classes < 2:
        raise ValueError("MemGuard requires at least two classes.")
    floor = min(confidence_floor, 1.0 - 1e-9)
    # Keep the top probability strictly above the uniform share of the rest so
    # the original argmax always survives, even for floors below 1/K.
    lower = max(floor, (1.0 / num_classes) * (1.0 + 1e-6))
    upper = 1.0 - 1e-9
    target = float(np.clip(entropy_target, 0.0, entropy_at_confidence(lower, num_classes)))
    for _ in range(int(bisection_iterations)):
        middle = 0.5 * (lower + upper)
        if entropy_at_confidence(middle, num_classes) > target:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def perturb_probabilities(
    probabilities: torch.Tensor,
    *,
    confidence_floor: float,
    entropy_target: float,
    bisection_iterations: int = 80,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perturb posteriors to the entropy target under the utility constraints.

    Every row is mapped to the canonical posterior ``[c, (1-c)/(K-1), ...]``
    placed on the row's original argmax, where ``c`` solves the entropy
    target under the confidence floor.  Rows already in that form are fixed
    points of the map.

    Args:
        probabilities: tensor of shape ``(batch, classes)`` on the simplex.
        confidence_floor: minimum kept probability of the original argmax
            class (the paper's ``0.5 + epsilon``).
        entropy_target: target entropy in nats; clipped to the feasible range.
        bisection_iterations: iterations of the scalar entropy bisection.

    Returns:
        ``(protected, stats)`` where ``protected`` holds the perturbed
        posteriors and ``stats`` summarizes the perturbation.
    """
    probs = probabilities.detach().to(torch.float32)
    if probs.ndim != 2:
        raise ValueError("probabilities must have shape (batch, classes).")
    num_rows, num_classes = probs.shape
    if num_classes < 2:
        raise ValueError("MemGuard requires at least two classes.")
    if not 0.0 < confidence_floor < 1.0:
        raise ValueError("confidence_floor must lie strictly between 0 and 1.")

    floor = min(confidence_floor, 1.0 - 1e-9)
    reachable_max = max_entropy_at_floor(num_classes, floor)
    target = float(np.clip(entropy_target, 0.0, reachable_max))
    if num_rows == 0:
        return probs.clone(), {
            "num_rows": 0,
            "num_classes": int(num_classes),
            "entropy_target": target,
            "entropy_floor_cap": reachable_max,
        }
    top_confidence = confidence_for_entropy(
        target, num_classes, floor, bisection_iterations
    )
    other_confidence = (1.0 - top_confidence) / (num_classes - 1)

    top = probs.argmax(dim=1)
    protected = torch.full_like(probs, other_confidence)
    protected[torch.arange(num_rows, device=probs.device), top] = top_confidence

    entropies = entropy_of_probabilities(protected)
    stats = {
        "num_rows": int(num_rows),
        "num_classes": int(num_classes),
        "entropy_target": target,
        "entropy_floor_cap": reachable_max,
        "top_confidence": float(top_confidence),
        "other_confidence": float(other_confidence),
        "max_abs_entropy_gap": float((entropies - target).abs().max().item()),
        "argmax_preserved_fraction": float(
            (protected.argmax(dim=1) == top).float().mean().item()
        ),
        "mean_abs_perturbation": float((protected - probs).abs().sum(dim=1).mean().item()),
    }
    return protected, stats


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, fpr_target: float) -> float:
    """TPR at a given FPR, rank-based and safe for tied scores."""
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1).astype(np.float64)
    positives = labels == 1
    n_pos = int(positives.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    sorted_scores = scores[order]
    cum_tp = np.cumsum(sorted_labels == 1)
    cum_fp = np.cumsum(sorted_labels == 0)
    # Thresholds are evaluated at the end of each block of tied scores.
    block_end = np.ones(len(scores), dtype=bool)
    block_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    fpr_values = cum_fp[block_end] / n_neg
    tpr_values = cum_tp[block_end] / n_pos
    feasible = fpr_values <= fpr_target + 1e-12
    return float(tpr_values[feasible].max()) if bool(feasible.any()) else 0.0


class _MemGuardProtectedPredictor(nn.Module):
    """Wrapper that releases protected posteriors as log-probabilities.

    Returning ``log(protected_probabilities)`` keeps the wrapper a drop-in
    logits-style classifier: ``softmax(output)`` reproduces the protected
    posterior exactly, so downstream metrics and attacks consume it unchanged.
    ``perturb`` is a frozen callable holding the resolved defense parameters,
    so later refits of the defense never change this predictor's behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        perturb: Any,
    ) -> None:
        super().__init__()
        self.model = model
        self.perturb = perturb

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            logits = extract_logits(self.model(samples))
        finally:
            self.model.train(was_training)
        probabilities = torch.softmax(logits, dim=1)
        protected, _ = self.perturb(probabilities)
        return protected.clamp_min(1e-12).log()


class MemGuardDefense(BaseDefense):
    """MemGuard inference-time posterior perturbation.

    defense_mode: inference_time

    Required:
        - ``target_model`` plus ``samples``, or precomputed
          ``signals['probabilities']`` / ``signals['logits']``
    Main output:
        - ``protected_outputs`` (perturbed probabilities) and
          ``protected_predictor`` (model wrapper, when a target model exists)

    ``fit`` calibrates the entropy target when non-member posteriors are
    available through ``auxiliary_data['nonmember_probabilities']`` or
    ``auxiliary_data['nonmember_data']`` (features pushed through the target
    model).  The paper calibrates with the non-member minimum entropy
    (``entropy_quantile = 0``); because a heavily over-fitted target is
    confident on non-members too, that minimum can collapse to ~0, so the
    robust default here is the median (``entropy_quantile = 0.5``).  Since
    the perturbation moves *every* posterior -- member or non-member -- onto
    the target from either side, both end up in the same entropy cluster and
    the exact quantile is not critical.  Without calibration data the target
    defaults to ``entropy_percentile_of_max`` of the maximum reachable
    entropy.
    """

    name = "memguard"
    defense_family = "output_perturbation"
    defense_mode = "inference_time"
    supported_model_types = ["classifier"]
    required_input_keys = [
        "target_model + samples, or signals['probabilities'] / signals['logits']"
    ]
    optional_input_keys = [
        "labels",
        "signals",
        "auxiliary_data.nonmember_probabilities",
        "auxiliary_data.nonmember_data",
        "train_data",
        "train_labels",
        "test_data",
        "test_labels",
        "defense_config",
        "eval_config",
    ]

    def __init__(
        self,
        confidence_floor: float = 0.51,
        entropy_target: Optional[float] = None,
        entropy_quantile: float = 0.5,
        entropy_percentile_of_max: float = 0.95,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.confidence_floor = float(confidence_floor)
        self.entropy_target = entropy_target
        self.entropy_quantile = float(entropy_quantile)
        self.entropy_percentile_of_max = float(entropy_percentile_of_max)
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.protected_predictor: Optional[nn.Module] = None
        self._effective_config: Dict[str, Any] = {}
        self._entropy_target: Optional[float] = None
        self._entropy_target_source: Optional[str] = None
        self._last_perturb_stats: Dict[str, Any] = {}
        self._last_perturb_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def fit(self, defense_input: DefenseInput) -> "MemGuardDefense":
        config = self._merge_config(defense_input.defense_config)
        self._effective_config = config
        self._entropy_target = None
        self._entropy_target_source = None
        # Reset per-call state so reusing one defense object never leaks the
        # previous call's model into a signals-only run.
        self.defended_model = None
        self.protected_predictor = None

        if defense_input.target_model is not None:
            if not isinstance(defense_input.target_model, nn.Module):
                raise TypeError("target_model must be a torch.nn.Module.")
            self.defended_model = defense_input.target_model.to(self.device)

        nonmember_probs = self._nonmember_probabilities(defense_input)
        if nonmember_probs is not None:
            if config["entropy_target"] is not None:
                self._set_entropy_target(float(config["entropy_target"]), "explicit")
            else:
                entropies = entropy_of_probabilities(nonmember_probs).numpy()
                quantile = float(np.quantile(entropies, config["entropy_quantile"]))
                cap = max_entropy_at_floor(
                    nonmember_probs.shape[1], config["confidence_floor"]
                )
                self._set_entropy_target(float(np.clip(quantile, 0.0, cap)), "calibrated_nonmember_quantile")
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        # Re-fit on every infer: fit is cheap and this keeps direct infer()
        # calls honest about the defense_config / calibration they carry.
        self.fit(defense_input)
        config = self._effective_config
        probs = self._input_probabilities(defense_input)
        num_classes = probs.shape[1]

        target = self._resolved_entropy_target(num_classes, config)
        floor = config["confidence_floor"]
        iterations = config["bisection_iterations"]

        start = time.perf_counter()
        protected, stats = perturb_probabilities(
            probs,
            confidence_floor=floor,
            entropy_target=target,
            bisection_iterations=iterations,
        )
        self._last_perturb_stats = stats
        self._last_perturb_seconds = time.perf_counter() - start

        if self.defended_model is not None:
            # The predictor freezes the resolved parameters so later refits of
            # this defense never retroactively change released predictors.
            frozen_perturb = self._frozen_perturb(num_classes)
            # No .eval() here: it would recurse into the wrapped user model
            # and flip its training mode permanently; forward() manages the
            # wrapped model's mode itself.
            self.protected_predictor = _MemGuardProtectedPredictor(
                self.defended_model, frozen_perturb
            ).to(self.device)

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.protected_predictor,
            protected_outputs=protected.numpy(),
            artifacts={
                "memguard_config": dict(config),
                "requested_entropy_target": self._entropy_target,
                "entropy_target": stats["entropy_target"],
                "entropy_target_source": self._entropy_target_source,
                "perturbation_stats": dict(stats),
            },
            intermediate_outputs={
                "raw_probabilities": probs.numpy(),
                "protected_probabilities": protected.numpy(),
                "raw_entropies": entropy_of_probabilities(probs).numpy(),
                "protected_entropies": entropy_of_probabilities(protected).numpy(),
            },
            metadata={
                "defense_name": self.name,
                "defense_family": self.defense_family,
                "defense_mode": self.defense_mode,
                "protected_output_type": "probabilities",
                "predictor_output_type": "log_probabilities",
                "requested_entropy_target": self._entropy_target,
                "entropy_target": stats["entropy_target"],
                "entropy_target_source": self._entropy_target_source,
                "top_confidence": stats.get("top_confidence"),
            },
        )

    def evaluate(
        self,
        defense_output: DefenseOutput,
        defense_input: DefenseInput,
    ) -> DefenseEvaluationResult:
        config = self._effective_config or self._merge_config(defense_input.defense_config)
        utility: Dict[str, float] = {}
        privacy: Dict[str, float] = {}
        intermediate = defense_output.intermediate_outputs or {}
        raw_probs = intermediate.get("raw_probabilities")
        protected_probs = intermediate.get("protected_probabilities")

        if raw_probs is not None and protected_probs is not None:
            utility["raw_mean_confidence"] = float(raw_probs.max(axis=1).mean())
            utility["protected_mean_confidence"] = float(protected_probs.max(axis=1).mean())
            privacy["raw_mean_entropy"] = float(
                entropy_of_probabilities(torch.as_tensor(raw_probs)).mean().item()
            )
            privacy["protected_mean_entropy"] = float(
                entropy_of_probabilities(torch.as_tensor(protected_probs)).mean().item()
            )
            if defense_input.labels is not None:
                labels = to_label_tensor(defense_input.labels)
                if len(labels) == len(raw_probs):
                    raw_tensor = torch.as_tensor(raw_probs)
                    protected_tensor = torch.as_tensor(protected_probs)
                    utility["raw_accuracy"] = float(
                        (raw_tensor.argmax(dim=1) == labels).float().mean().item()
                    )
                    utility["protected_accuracy"] = float(
                        (protected_tensor.argmax(dim=1) == labels).float().mean().item()
                    )
                    utility["prediction_preservation"] = float(
                        (raw_tensor.argmax(dim=1) == protected_tensor.argmax(dim=1))
                        .float()
                        .mean()
                        .item()
                    )

        predictor = defense_output.protected_predictor
        has_train = defense_input.train_data is not None and defense_input.train_labels is not None
        has_test = defense_input.test_data is not None and defense_input.test_labels is not None
        if predictor is not None and self.defended_model is not None and has_train and has_test:
            batch_size = config["batch_size"]
            for prefix, data, labels in (
                ("train", defense_input.train_data, defense_input.train_labels),
                ("test", defense_input.test_data, defense_input.test_labels),
            ):
                metrics = classifier_metrics(
                    predictor, data, labels, device=self.device, batch_size=batch_size
                )
                utility.update({f"{prefix}_{key}": value for key, value in metrics.items()})

            # Score the exact predictor shipped in this DefenseOutput (its
            # output is log protected-probabilities, so softmax recovers the
            # protected posterior exactly).
            raw_member = torch.softmax(
                predict_logits(
                    self.defended_model,
                    defense_input.train_data,
                    device=self.device,
                    batch_size=batch_size,
                ),
                dim=1,
            )
            raw_nonmember = torch.softmax(
                predict_logits(
                    self.defended_model,
                    defense_input.test_data,
                    device=self.device,
                    batch_size=batch_size,
                ),
                dim=1,
            )
            protected_member = torch.softmax(
                predict_logits(
                    predictor,
                    defense_input.train_data,
                    device=self.device,
                    batch_size=batch_size,
                ),
                dim=1,
            )
            protected_nonmember = torch.softmax(
                predict_logits(
                    predictor,
                    defense_input.test_data,
                    device=self.device,
                    batch_size=batch_size,
                ),
                dim=1,
            )
            privacy.update(
                self._privacy_metrics(
                    raw_member,
                    raw_nonmember,
                    protected_member,
                    protected_nonmember,
                    defense_input.train_labels,
                    defense_input.test_labels,
                )
            )
        elif raw_probs is not None and protected_probs is not None:
            membership = (defense_input.eval_config or {}).get("membership_labels")
            if membership is not None:
                membership_labels = to_label_tensor(membership).numpy()
                split = self._split_by_membership(
                    raw_probs, protected_probs, membership_labels, defense_input.labels
                )
                if split is not None:
                    privacy.update(self._privacy_metrics(*split))

        efficiency: Dict[str, float] = {}
        if self._last_perturb_seconds is not None:
            efficiency["perturb_seconds"] = float(self._last_perturb_seconds)
            rows = self._last_perturb_stats.get("num_rows")
            if rows:
                efficiency["perturb_seconds_per_sample"] = float(
                    self._last_perturb_seconds / rows
                )

        return DefenseEvaluationResult(
            utility_metrics=utility or None,
            privacy_metrics=privacy or None,
            efficiency_metrics=efficiency or None,
            extra_metrics={"perturbation_stats": dict(self._last_perturb_stats)},
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _merge_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "confidence_floor": self.confidence_floor,
            "entropy_target": self.entropy_target,
            "entropy_quantile": self.entropy_quantile,
            "entropy_percentile_of_max": self.entropy_percentile_of_max,
            "bisection_iterations": 80,
            "batch_size": self.batch_size,
        }
        config.update(dict(overrides or {}))
        for key in (
            "confidence_floor",
            "entropy_quantile",
            "entropy_percentile_of_max",
        ):
            config[key] = float(config[key])
        for key in ("bisection_iterations", "batch_size"):
            config[key] = int(config[key])
        if config["entropy_target"] is not None:
            config["entropy_target"] = float(config["entropy_target"])
        if not 0.0 < config["confidence_floor"] < 1.0:
            raise ValueError("confidence_floor must lie strictly between 0 and 1.")
        if not 0.0 <= config["entropy_quantile"] <= 1.0:
            raise ValueError("entropy_quantile must lie in [0, 1].")
        if not 0.0 <= config["entropy_percentile_of_max"] <= 1.0:
            raise ValueError("entropy_percentile_of_max must lie in [0, 1].")
        if config["bisection_iterations"] <= 0 or config["batch_size"] <= 0:
            raise ValueError("bisection_iterations and batch_size must be positive.")
        return config

    def _set_entropy_target(self, value: float, source: str) -> None:
        self._entropy_target = float(value)
        self._entropy_target_source = source

    def _resolved_entropy_target(self, num_classes: int, config: Dict[str, Any]) -> float:
        """Entropy target for ad-hoc perturbations, lazily resolved and clipped."""
        if self._entropy_target is None:
            if config["entropy_target"] is not None:
                self._set_entropy_target(float(config["entropy_target"]), "explicit")
            else:
                cap = max_entropy_at_floor(num_classes, config["confidence_floor"])
                self._set_entropy_target(
                    config["entropy_percentile_of_max"] * cap, "percentile_of_max"
                )
        cap = max_entropy_at_floor(num_classes, config["confidence_floor"])
        return float(np.clip(self._entropy_target, 0.0, cap))

    def _nonmember_probabilities(self, defense_input: DefenseInput) -> Optional[torch.Tensor]:
        auxiliary = defense_input.auxiliary_data or {}
        nonmember_probs = auxiliary.get("nonmember_probabilities")
        if nonmember_probs is not None:
            probs = torch.as_tensor(nonmember_probs, dtype=torch.float32).detach()
            if probs.ndim != 2:
                raise ValueError("auxiliary_data['nonmember_probabilities'] must be 2-D.")
            return probs
        nonmember_data = auxiliary.get("nonmember_data")
        if nonmember_data is not None and self.defended_model is not None:
            logits = predict_logits(
                self.defended_model,
                nonmember_data,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )
            return torch.softmax(logits, dim=1)
        return None

    def _input_probabilities(self, defense_input: DefenseInput) -> torch.Tensor:
        signals = defense_input.signals or {}
        if signals.get("probabilities") is not None:
            probs = torch.as_tensor(signals["probabilities"], dtype=torch.float32).detach()
        elif signals.get("logits") is not None:
            logits = torch.as_tensor(signals["logits"], dtype=torch.float32).detach()
            probs = torch.softmax(logits, dim=1)
        elif defense_input.samples is not None and self.defended_model is not None:
            logits = predict_logits(
                self.defended_model,
                defense_input.samples,
                device=self.device,
                batch_size=self._effective_config["batch_size"],
            )
            probs = torch.softmax(logits, dim=1)
        else:
            raise ValueError(
                "MemGuardDefense requires target_model plus samples, or "
                "signals['probabilities'] / signals['logits']."
            )
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("MemGuard requires posteriors of shape (batch, classes >= 2).")
        return probs

    def _frozen_perturb(self, num_classes: int) -> Any:
        """Perturbation callable with the resolved parameters baked in."""
        floor = self._effective_config["confidence_floor"]
        target = self._resolved_entropy_target(num_classes, self._effective_config)
        iterations = self._effective_config["bisection_iterations"]

        def perturb(probabilities: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
            return perturb_probabilities(
                probabilities,
                confidence_floor=floor,
                entropy_target=target,
                bisection_iterations=iterations,
            )

        return perturb

    def _split_by_membership(
        self,
        raw_probs: np.ndarray,
        protected_probs: np.ndarray,
        membership_labels: np.ndarray,
        task_labels: Any,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, Any]]:
        if len(membership_labels) != len(raw_probs):
            return None
        member_mask = membership_labels == 1
        nonmember_mask = ~member_mask
        if not member_mask.any() or not nonmember_mask.any():
            return None
        raw_tensor = torch.as_tensor(raw_probs)
        protected_tensor = torch.as_tensor(protected_probs)
        if task_labels is not None and len(to_label_tensor(task_labels)) == len(raw_probs):
            targets = to_label_tensor(task_labels).numpy()
            member_targets: Optional[np.ndarray] = targets[member_mask]
            nonmember_targets: Optional[np.ndarray] = targets[nonmember_mask]
        else:
            member_targets = None
            nonmember_targets = None
        return (
            raw_tensor[member_mask],
            raw_tensor[nonmember_mask],
            protected_tensor[member_mask],
            protected_tensor[nonmember_mask],
            member_targets,
            nonmember_targets,
        )

    def _privacy_metrics(
        self,
        raw_member: torch.Tensor,
        raw_nonmember: torch.Tensor,
        protected_member: torch.Tensor,
        protected_nonmember: torch.Tensor,
        member_labels: Any,
        nonmember_labels: Any,
    ) -> Dict[str, float]:
        """AUROC / TPR@1%FPR of confidence, entropy and loss attacks, before vs after."""
        membership = np.concatenate(
            [
                np.ones(len(raw_member), dtype=np.int64),
                np.zeros(len(raw_nonmember), dtype=np.int64),
            ]
        )
        member_targets = (
            to_label_tensor(member_labels).numpy() if member_labels is not None else None
        )
        nonmember_targets = (
            to_label_tensor(nonmember_labels).numpy() if nonmember_labels is not None else None
        )

        def signal_scores(
            probs: torch.Tensor, targets: Optional[np.ndarray]
        ) -> Dict[str, np.ndarray]:
            scores = {
                "confidence": probs.max(dim=1).values.numpy(),
                "entropy": -entropy_of_probabilities(probs).numpy(),
            }
            if targets is not None:
                rows = np.arange(len(targets))
                scores["loss"] = probs.clamp_min(1e-12).log()[rows, targets].numpy()
            return scores

        raw_scores = signal_scores(raw_member, member_targets)
        raw_scores.update(
            {f"nonmember_{k}": v for k, v in signal_scores(raw_nonmember, nonmember_targets).items()}
        )
        protected_scores = signal_scores(protected_member, member_targets)
        protected_scores.update(
            {
                f"nonmember_{k}": v
                for k, v in signal_scores(protected_nonmember, nonmember_targets).items()
            }
        )

        metrics: Dict[str, float] = {}
        raw_aurocs = []
        protected_aurocs = []
        signals = ("confidence", "entropy") + (("loss",) if member_targets is not None else ())
        for signal in signals:
            raw_signal = np.concatenate([raw_scores[signal], raw_scores[f"nonmember_{signal}"]])
            protected_signal = np.concatenate(
                [protected_scores[signal], protected_scores[f"nonmember_{signal}"]]
            )
            raw_auroc = binary_auroc(membership, raw_signal)
            protected_auroc = binary_auroc(membership, protected_signal)
            raw_aurocs.append(raw_auroc)
            protected_aurocs.append(protected_auroc)
            metrics[f"attack_{signal}_auroc_raw"] = raw_auroc
            metrics[f"attack_{signal}_auroc_protected"] = protected_auroc
            metrics[f"privacy_gain_{signal}"] = raw_auroc - protected_auroc
            metrics[f"attack_{signal}_tpr_at_1pct_fpr_raw"] = tpr_at_fpr(
                membership, raw_signal, 0.01
            )
            metrics[f"attack_{signal}_tpr_at_1pct_fpr_protected"] = tpr_at_fpr(
                membership, protected_signal, 0.01
            )
        metrics["clean_attack_auroc"] = max(raw_aurocs)
        metrics["defended_attack_auroc"] = max(protected_aurocs)
        metrics["privacy_gain"] = metrics["clean_attack_auroc"] - metrics["defended_attack_auroc"]
        return metrics


__all__ = [
    "MemGuardDefense",
    "confidence_for_entropy",
    "entropy_at_confidence",
    "entropy_of_probabilities",
    "max_entropy_at_floor",
    "perturb_probabilities",
    "tpr_at_fpr",
]
