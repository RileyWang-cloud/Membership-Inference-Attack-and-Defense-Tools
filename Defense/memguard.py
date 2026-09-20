"""MemGuard output-perturbation defense for PyTorch classifiers.

Reference: Jia, Gong, Bi, Li and Cao, "MemGuard: Defending against Black-Box
Membership Inference Attacks via Adversarial Examples", CCS 2019.
Official implementation: https://github.com/jinyuan-jia/MemGuard
(local snapshot: ``Ref/MemGuard_official/``, in particular
``defense_framework.py`` and ``train_defense_model_defensemodel.py``).

This file follows the official method closely:

1. ``fit`` trains the defender's own surrogate attack model -- a small binary
   classifier that separates the *sorted posteriors* of the target model on
   its training data (members, label 1) from posteriors on non-member data
   (label 0).  Architecture and objective mirror the official
   ``model_defense``: Dense(256)-relu -> Dense(128)-relu -> Dense(64)-relu ->
   Dense(1) with sigmoid, trained with binary cross-entropy (400 epochs,
   batch 64).  Two training deviations, both motivated below: the surrogate
   is optimized with Adam instead of the official SGD (lr 0.001) -- those
   constants majority-collapse on calibration sets smaller than the official
   2000 posteriors of a 30-class model, while Adam converges across scales --
   and the training is repeated from several independent initializations.

2. ``infer`` perturbs every released posterior with the official adversarial
   optimization.  Working in sorted-posterior coordinates on the logit vector
   ``f`` (posterior = softmax(f), so gradients flow through the softmax just
   like the official ``model_defense_optimize``), each step minimizes the
   Lagrangian

       c1 * |surrogate_logit(softmax(f))|                   (privacy)
       + c2 * relu(max_other(f) - f[predicted])              (keep argmax)
       + c3 * || softmax(f) - original_posterior ||_1        (stay close)

   with the official constants ``c1 = 1.0``, ``c2 = 10.0``, ``c3 = 0.1``
   growing 10x per outer round up to 1e5, a step size of 0.1 along the L2
   normalized gradient, and at most 300 inner iterations.  The inner loop
   stops once the predicted label is intact and the surrogate score has
   reached the 0.5 decision boundary; a sample keeps the successful
   perturbation that lands closest to the boundary, or the original posterior
   if the optimization fails outright -- so the argmax prediction, and
   therefore accuracy, is preserved exactly.

Deviations from the official code, all documented here and in the function
defaults:

- mechanical: the per-sample TensorFlow loop is re-expressed as a batched
  PyTorch state machine with the same per-sample semantics (same losses,
  constants, resets, and stopping rules), which makes the defense vectorized
  over the query batch;
- the official inner loop stops only when the surrogate score strictly
  crosses 0.5 to the other side.  With a finite step size the iterate can
  instead stall asymptotically just short of the boundary (observed at
  |score - 0.5| ~ 0.004), and the official rule would then leave those
  posteriors undefended.  This implementation therefore also accepts
  ``|score - 0.5| <= crossing_tolerance`` (default 0.01) as having reached
  the boundary, which is the paper's stated goal (drive the attack model's
  output to 0.5); the official 1e-5 tolerance on the *initial* score is kept
  for the already-calibrated early exit;
- a single step of finite size can also *overshoot*: the score jumps past
  0.5 to the other side and the official stopping rule accepts it regardless
  of magnitude.  Overshooting steps are refined by bisecting the step
  segment onto the 0.5 boundary, and when several outer rounds succeed the
  perturbation landing closest to the boundary is kept -- both serve the
  paper's stated goal;
- the surrogate is trained ``surrogate_restarts`` (default 3) times from
  independent initializations, keeping the run with the lowest final training
  loss.  On calibration sets far smaller than the official 2000 posteriors a
  single initialization can land in a basin whose boundary barely separates
  members from non-members, and the perturbation then stalls far from 0.5;
  retraining from a different seed reliably escapes such basins.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

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

# Official tolerance on the *initial* surrogate score: a posterior whose
# score already sits this close to the 0.5 boundary is released untouched.
_ALREADY_CALIBRATED_TOLERANCE = 1e-5


# ----------------------------------------------------------------------
# Surrogate attack model (official ``model_defense`` / ``model_defense_optimize``)
# ----------------------------------------------------------------------


class _SurrogateAttackNet(nn.Module):
    """Official ``model_defense`` stack: posteriors -> 256 -> 128 -> 64 -> 1.

    The official ``model_defense_optimize`` used inside the perturbation loop
    is this same network with a softmax spliced in front of the input; here
    the softmax lives at the call sites so one set of weights serves both
    roles, exactly like the official weight sharing.
    """

    def __init__(self, num_classes: int, hidden_dims: Sequence[int] = (256, 128, 64)) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        previous = int(num_classes)
        for width in hidden_dims:
            layers += [nn.Linear(previous, int(width)), nn.ReLU()]
            previous = int(width)
        layers += [nn.Linear(previous, 1)]
        self.network = nn.Sequential(*layers)

    def forward(self, posteriors: torch.Tensor) -> torch.Tensor:
        """Membership logit (pre-sigmoid); shape ``(batch,)``."""
        return self.network(posteriors).squeeze(1)


def train_surrogate_attack_model(
    member_posteriors: torch.Tensor,
    nonmember_posteriors: torch.Tensor,
    *,
    hidden_dims: Sequence[int] = (256, 128, 64),
    epochs: int = 400,
    learning_rate: float = 0.005,
    batch_size: int = 64,
    seed: int = 1000,
    restarts: int = 3,
    device: Optional[str] = None,
) -> Tuple[_SurrogateAttackNet, Dict[str, Any]]:
    """Train the surrogate membership classifier on sorted posteriors.

    Mirrors ``train_defense_model_defensemodel.py``: features are the target
    model's posteriors sorted per row, labels are 1 for members and 0 for
    non-members, trained on binary cross-entropy.  Two robustness deviations
    from the official single SGD (lr 0.001) run, both documented in the
    module docstring: the optimizer is Adam (official constants stall in the
    majority class on small calibration sets), and `restarts` independent
    initializations are trained, keeping the one with the lowest final
    training loss (on small calibration sets a single random init can land
    in a basin whose decision boundary barely separates the classes).
    """
    members = torch.as_tensor(member_posteriors, dtype=torch.float32).detach()
    nonmembers = torch.as_tensor(nonmember_posteriors, dtype=torch.float32).detach()
    if members.ndim != 2 or nonmembers.ndim != 2:
        raise ValueError("member/nonmember posteriors must have shape (batch, classes).")
    if members.shape[1] != nonmembers.shape[1]:
        raise ValueError("member and nonmember posteriors must share the class dimension.")
    if len(members) == 0 or len(nonmembers) == 0:
        raise ValueError("MemGuard needs at least one member and one non-member posterior.")

    # Official trains on the sorted posteriors (np.sort along the class axis).
    features = torch.sort(torch.cat([members, nonmembers]), dim=1).values.to(device)
    labels = torch.cat(
        [torch.ones(len(members)), torch.zeros(len(nonmembers))]
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    num_rows = len(features)

    best_surrogate: Optional[_SurrogateAttackNet] = None
    best_loss: Optional[float] = None
    info: Dict[str, Any] = {}
    restart_losses: List[float] = []
    restart_train_accuracies: List[float] = []
    for restart in range(max(1, int(restarts))):
        restart_seed = int(seed) + restart
        # fork_rng only takes CUDA device ids for the extra-device argument;
        # the CPU generator is always forked, which is all this init needs.
        with torch.random.fork_rng():
            torch.manual_seed(restart_seed)
            surrogate = _SurrogateAttackNet(features.shape[1], hidden_dims).to(device)
        surrogate.train()
        optimizer = torch.optim.Adam(surrogate.parameters(), lr=float(learning_rate))
        generator = torch.Generator().manual_seed(restart_seed)

        for _ in range(int(epochs)):
            order = torch.randperm(num_rows, generator=generator).to(device)
            for start in range(0, num_rows, int(batch_size)):
                batch = order[start : start + int(batch_size)]
                logits = surrogate(features[batch])
                loss = criterion(logits, labels[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        surrogate.eval()
        with torch.no_grad():
            final_logits = surrogate(features)
            final_loss = float(criterion(final_logits, labels))
            final_accuracy = float(((final_logits > 0.0) == labels).float().mean())
        restart_losses.append(final_loss)
        restart_train_accuracies.append(final_accuracy)
        if best_loss is None or final_loss < best_loss:
            best_loss = final_loss
            best_surrogate = surrogate
            info = {
                "num_member_posteriors": int(len(members)),
                "num_nonmember_posteriors": int(len(nonmembers)),
                "num_classes": int(features.shape[1]),
                "hidden_dims": [int(width) for width in hidden_dims],
                "epochs": int(epochs),
                "learning_rate": float(learning_rate),
                "batch_size": int(batch_size),
                "seed": int(restart_seed),
                "restarts": max(1, int(restarts)),
                "restart_index": restart,
                "train_loss": final_loss,
                "train_accuracy": final_accuracy,
            }
    assert best_surrogate is not None
    info["restart_losses"] = list(restart_losses)
    info["restart_train_accuracies"] = list(restart_train_accuracies)
    return best_surrogate, info


# ----------------------------------------------------------------------
# Perturbation (official ``defense_framework.py`` loop, batched)
# ----------------------------------------------------------------------


def perturb_posteriors(
    probabilities: torch.Tensor,
    surrogate: _SurrogateAttackNet,
    *,
    c1: float = 1.0,
    c2: float = 10.0,
    c3_init: float = 0.1,
    c3_growth: float = 10.0,
    c3_max: float = 100000.0,
    step_size: float = 0.1,
    max_iterations: int = 300,
    crossing_tolerance: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Perturb posteriors so the surrogate attack model cannot separate them.

    Vectorized equivalent of the official per-sample loop in
    ``defense_framework.py``; every constant keeps its official value by
    default.  See the module docstring for the Lagrangian and the stopping
    rules.  ``crossing_tolerance`` widens the official strict-crossing stop
    to "within tolerance of the 0.5 boundary" (module docstring, deviation 2).
    Samples whose optimization fails keep their original posterior,
    and the argmax prediction is preserved for every row either way.  When
    several outer rounds succeed, the perturbation landing closest to the
    0.5 boundary is kept (module docstring, deviation 3).

    Returns ``(protected, stats)``.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, classes).")
    if probabilities.shape[1] < 2:
        raise ValueError("MemGuard requires at least two classes.")
    # Work on the surrogate's device; the optimization below must also run
    # under enable_grad so it works when the caller wrapped us in no_grad
    # (e.g. inside the protected predictor's forward).
    device = next(surrogate.parameters()).device
    probs = probabilities.detach().to(device=device, dtype=torch.float32)
    num_rows, num_classes = probs.shape
    if num_rows == 0:
        return probs.clone(), {
            "num_rows": 0,
            "num_classes": int(num_classes),
            "perturbed_fraction": 0.0,
            "kept_original_fraction": 0.0,
            "already_calibrated_fraction": 0.0,
            "label_failure_fraction": 0.0,
            "maxiter_failure_fraction": 0.0,
            "mean_outer_rounds_perturbed": 0.0,
            "mean_inner_steps_perturbed": 0.0,
            "mean_l1_perturbation": 0.0,
            "surrogate_score_gap_before": 0.0,
            "surrogate_score_gap_after": 0.0,
            "argmax_preserved_fraction": 0.0,
        }

    # Stable sort keeps tied maxima in their original column order, so the
    # argmax position in sorted coordinates always maps back to the column
    # probs.argmax would report (torch.sort is not stable by default).
    sorted_probs, sort_idx = torch.sort(probs, dim=1, stable=True)
    # The official loop optimizes the logit vector whose softmax is the
    # posterior; from posteriors alone the monotone stand-in is the
    # log-posterior (its softmax reproduces the posterior exactly).
    origin_logits = sorted_probs.clamp_min(1e-12).log()
    # One-hot of the predicted position in sorted coordinates (ascending, so
    # this is the last position unless the maximum is tied).
    max_pos = sorted_probs.argmax(dim=1)
    label_mask = torch.nn.functional.one_hot(max_pos, num_classes).to(torch.float32)

    was_training = surrogate.training
    surrogate.eval()
    try:
        with torch.enable_grad():
            with torch.no_grad():
                initial_scores = torch.sigmoid(surrogate(sorted_probs))
            initial_side = initial_scores - 0.5

            best = sorted_probs.clone()
            best_scores = initial_scores.clone()
            ever_perturbed = torch.zeros(num_rows, dtype=torch.bool, device=device)
            # 0 = none, 1 = label failure, 2 = max-iteration failure.
            fail_code = torch.zeros(num_rows, dtype=torch.long, device=device)
            # Official early exit: a score within 1e-5 of the boundary is
            # already calibrated and keeps its posterior untouched.  This is
            # deliberately NOT crossing_tolerance, which only governs the
            # perturbation stop (module docstring, deviation 2).
            already = initial_side.abs() <= _ALREADY_CALIBRATED_TOLERANCE
            finished = already.clone()
            c3 = torch.full((num_rows,), float(c3_init), device=device)
            rounds_used = torch.zeros(num_rows, dtype=torch.long, device=device)
            steps_used = torch.zeros(num_rows, dtype=torch.long, device=device)
            # Score gap of the current iterate (starts at the origin); lets
            # the overshoot refinement below verify its bracket.
            current_gap = initial_side.clone()

            while bool((~finished).any()):
                active = ~finished
                # Official resets the iterate to the original logits every outer
                # round before re-running the inner loop.
                sample_f = origin_logits.clone()
                stepping = active.clone()

                for _ in range(int(max_iterations)):
                    if not bool(stepping.any()):
                        break
                    iterate = sample_f.detach().requires_grad_(True)
                    probs_f = torch.softmax(iterate, dim=1)
                    score_logit = surrogate(probs_f)
                    correct = (label_mask * iterate).sum(dim=1)
                    wrong = (iterate - 1e8 * label_mask).max(dim=1).values
                    per_row_loss = (
                        c1 * score_logit.abs()
                        + c2 * torch.relu(wrong - correct)
                        + c3 * (probs_f - sorted_probs).abs().sum(dim=1)
                    )
                    # Rows outside `stepping` do not contribute, so their
                    # gradients are exactly zero and their iterates stay frozen.
                    gradient = torch.autograd.grad(per_row_loss[stepping].sum(), iterate)[0]
                    gradient = gradient / gradient.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    with torch.no_grad():
                        previous_f = sample_f
                        sample_f = sample_f - step_size * gradient
                        scores = torch.sigmoid(surrogate(torch.softmax(sample_f, dim=1)))
                        gap = scores - 0.5
                        # A step of finite size can overshoot: the score jumps
                        # past 0.5 to the other side, far outside tolerance.
                        # The score is continuous along the step segment and
                        # the previous iterate was still on the origin side
                        # (checked via current_gap), so bisecting the segment
                        # lands on the 0.5 boundary (the paper's stated goal).
                        # Rows that already sit past the boundary -- possible
                        # while the argmax constraint is still being repaired
                        # -- have no valid bracket and keep the stepped
                        # iterate, exactly like the official loop.
                        overshoot = (
                            stepping
                            & (gap * initial_side < 0.0)
                            & (gap.abs() > crossing_tolerance)
                            & (current_gap * initial_side > 0.0)
                        )
                        if bool(overshoot.any()):
                            indices = overshoot.nonzero(as_tuple=True)[0]
                            lo = previous_f[indices]
                            hi = sample_f[indices]
                            hi_gap = gap[indices]
                            for _ in range(10):
                                mid = 0.5 * (lo + hi)
                                mid_gap = (
                                    torch.sigmoid(surrogate(torch.softmax(mid, dim=1))) - 0.5
                                )
                                crossed_side = mid_gap * hi_gap > 0.0
                                hi = torch.where(crossed_side[:, None], mid, hi)
                                lo = torch.where(crossed_side[:, None], lo, mid)
                            lo_gap = (
                                torch.sigmoid(surrogate(torch.softmax(lo, dim=1))) - 0.5
                            )
                            hi_gap = (
                                torch.sigmoid(surrogate(torch.softmax(hi, dim=1))) - 0.5
                            )
                            take_hi = hi_gap.abs() < lo_gap.abs()
                            refined = torch.where(take_hi[:, None], hi, lo)
                            refined_gap = torch.where(take_hi, hi_gap, lo_gap)
                            sample_f = sample_f.index_copy(0, indices, refined)
                            gap = gap.index_copy(0, indices, refined_gap)
                        current_gap = gap
                        argmax = sample_f.argmax(dim=1)
                        crossed = (gap * initial_side <= 0.0) | (gap.abs() <= crossing_tolerance)
                        steps_used += stepping.to(torch.long)
                    # Inner-loop stop: predicted label intact AND surrogate score
                    # reached the 0.5 boundary (crossed it, or within tolerance).
                    stepping &= ~((argmax == max_pos) & crossed)

                with torch.no_grad():
                    final_scores = torch.sigmoid(surrogate(torch.softmax(sample_f, dim=1)))
                    final_argmax = sample_f.argmax(dim=1)
                    final_gap = final_scores - 0.5
                    final_crossed = (final_gap * initial_side <= 0.0) | (
                        final_gap.abs() <= crossing_tolerance
                    )
                # Post-loop checks in the official order: argmax first, then
                # whether the score ever crossed 0.5.
                label_failed = active & (final_argmax != max_pos)
                maxiter_failed = active & ~label_failed & ~final_crossed
                success = active & ~label_failed & ~maxiter_failed

                new_best = torch.softmax(sample_f, dim=1)
                # The official code keeps the last success of the c3-growth
                # loop; a single step of finite size can also overshoot 0.5 by
                # a wide margin (sign flip with |score - 0.5| >> tolerance),
                # so among the successful rounds we keep the one landing
                # closest to the 0.5 boundary -- the paper's stated goal.
                take = success & (~ever_perturbed | (final_gap.abs() < (best_scores - 0.5).abs()))
                best = torch.where(take[:, None], new_best, best)
                best_scores = torch.where(take, final_scores, best_scores)
                ever_perturbed |= take
                fail_code[label_failed] = 1
                fail_code[maxiter_failed] = 2
                rounds_used[active] += 1
                c3 = torch.where(success, c3 * float(c3_growth), c3)
                finished |= label_failed | maxiter_failed | (c3 > float(c3_max))
    finally:
        surrogate.train(was_training)

    protected = torch.empty_like(best)
    protected.scatter_(1, sort_idx, best)
    # Hand the result back on the caller's device so downstream .numpy()
    # conversions work regardless of where the surrogate lives.
    protected = protected.to(probabilities.device)

    stats = {
        "num_rows": int(num_rows),
        "num_classes": int(num_classes),
        "perturbed_fraction": float(ever_perturbed.float().mean()),
        "kept_original_fraction": float((~ever_perturbed).float().mean()),
        "already_calibrated_fraction": float(already.float().mean()),
        "label_failure_fraction": float((fail_code == 1).float().mean()),
        "maxiter_failure_fraction": float((fail_code == 2).float().mean()),
        "mean_outer_rounds_perturbed": float(
            rounds_used[ever_perturbed].float().mean() if bool(ever_perturbed.any()) else 0.0
        ),
        "mean_inner_steps_perturbed": float(
            steps_used[ever_perturbed].float().mean() if bool(ever_perturbed.any()) else 0.0
        ),
        "mean_l1_perturbation": float((protected - probs).abs().sum(dim=1).mean()),
        "surrogate_score_gap_before": float((initial_scores - 0.5).abs().mean()),
        "surrogate_score_gap_after": float((best_scores - 0.5).abs().mean()),
        "argmax_preserved_fraction": float(
            (protected.argmax(dim=1) == probs.argmax(dim=1)).float().mean()
        ),
    }
    return protected, stats


def entropy_of_probabilities(probabilities: torch.Tensor) -> torch.Tensor:
    """Natural entropy of each row of a probability matrix.

    Sibling of ``_classification.probability_entropy`` (which softmaxes a
    logits matrix first); keep the two formulas in sync.
    """
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (batch, classes).")
    clamped = probabilities.clamp_min(1e-12)
    return -(probabilities * clamped.log()).sum(dim=1)


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
    The surrogate and solver parameters are captured at construction time, so
    later refits of the defense never change this predictor's behavior.
    """

    def __init__(
        self,
        model: nn.Module,
        surrogate: _SurrogateAttackNet,
        solver_config: Dict[str, Any],
    ) -> None:
        super().__init__()
        self.model = model
        self.surrogate = surrogate
        self.solver_config = dict(solver_config)

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        was_training = self.model.training
        self.model.eval()
        try:
            logits = extract_logits(self.model(samples))
        finally:
            self.model.train(was_training)
        probabilities = torch.softmax(logits, dim=1)
        protected, _ = perturb_posteriors(probabilities, self.surrogate, **self.solver_config)
        return protected.clamp_min(1e-12).log()


class MemGuardDefense(BaseDefense):
    """MemGuard inference-time posterior perturbation (official algorithm).

    defense_mode: inference_time

    Required (to train the surrogate attack model and to perturb):
        - member posteriors: ``auxiliary_data['member_probabilities']``, or
          ``target_model`` + ``train_data``
        - non-member posteriors: ``auxiliary_data['nonmember_probabilities']``,
          or ``target_model`` + ``auxiliary_data['nonmember_data']`` / ``test_data``
        - posteriors to protect: ``target_model`` + ``samples``, or
          ``signals['probabilities']`` / ``signals['logits']``
    Main output:
        - ``protected_outputs`` (perturbed probabilities) and
          ``protected_predictor`` (model wrapper, when a target model exists)

    The surrogate and the perturbation solver keep the official MemGuard
    hyperparameters by default; every one of them can be overridden through
    ``defense_config``.  The defense preserves each row's argmax prediction
    exactly (failed optimizations keep the original posterior), so task
    accuracy is unchanged by construction.
    """

    name = "memguard"
    defense_family = "output_perturbation"
    defense_mode = "inference_time"
    supported_model_types = ["classifier"]
    required_input_keys = [
        "member posteriors + nonmember posteriors + posteriors to protect "
        "(via target_model/data or auxiliary_data/signals)"
    ]
    optional_input_keys = [
        "labels",
        "signals",
        "auxiliary_data.member_probabilities",
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
        surrogate_hidden_dims: Sequence[int] = (256, 128, 64),
        surrogate_epochs: int = 400,
        surrogate_learning_rate: float = 0.005,
        surrogate_batch_size: int = 64,
        surrogate_seed: int = 1000,
        surrogate_restarts: int = 3,
        c1: float = 1.0,
        c2: float = 10.0,
        c3_init: float = 0.1,
        c3_growth: float = 10.0,
        c3_max: float = 100000.0,
        step_size: float = 0.1,
        max_iterations: int = 300,
        crossing_tolerance: float = 0.01,
        batch_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.surrogate_hidden_dims = tuple(int(width) for width in surrogate_hidden_dims)
        self.surrogate_epochs = int(surrogate_epochs)
        self.surrogate_learning_rate = float(surrogate_learning_rate)
        self.surrogate_batch_size = int(surrogate_batch_size)
        self.surrogate_seed = int(surrogate_seed)
        self.surrogate_restarts = int(surrogate_restarts)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3_init = float(c3_init)
        self.c3_growth = float(c3_growth)
        self.c3_max = float(c3_max)
        self.step_size = float(step_size)
        self.max_iterations = int(max_iterations)
        self.crossing_tolerance = float(crossing_tolerance)
        self.batch_size = int(batch_size)
        self.device = resolve_device(device)

        self.defended_model: Optional[nn.Module] = None
        self.protected_predictor: Optional[nn.Module] = None
        self._surrogate: Optional[_SurrogateAttackNet] = None
        self._effective_config: Dict[str, Any] = {}
        self._fit_info: Dict[str, Any] = {}
        self._last_perturb_stats: Dict[str, Any] = {}
        self._last_perturb_seconds: Optional[float] = None

    # ------------------------------------------------------------------
    # BaseDefense interface
    # ------------------------------------------------------------------

    def fit(self, defense_input: DefenseInput) -> "MemGuardDefense":
        config = self._merge_config(defense_input.defense_config)
        self._effective_config = config
        self._fit_info = {}
        # Reset per-call state so reusing one defense object never leaks the
        # previous call's model into a signals-only run.
        self.defended_model = None
        self.protected_predictor = None
        self._surrogate = None

        if defense_input.target_model is not None:
            if not isinstance(defense_input.target_model, nn.Module):
                raise TypeError("target_model must be a torch.nn.Module.")
            self.defended_model = defense_input.target_model.to(self.device)

        member_probs = self._member_probabilities(defense_input)
        nonmember_probs = self._nonmember_probabilities(defense_input)
        if member_probs is None or nonmember_probs is None:
            missing = []
            if member_probs is None:
                missing.append("member posteriors (auxiliary_data['member_probabilities'] or target_model + train_data)")
            if nonmember_probs is None:
                missing.append(
                    "non-member posteriors (auxiliary_data['nonmember_probabilities'] / "
                    "['nonmember_data'], or target_model + test_data)"
                )
            raise ValueError(
                "MemGuard trains a surrogate attack model on member vs non-member "
                "posteriors and cannot run without them; missing: " + "; ".join(missing) + "."
            )

        surrogate, info = train_surrogate_attack_model(
            member_probs,
            nonmember_probs,
            hidden_dims=tuple(config["surrogate_hidden_dims"]),
            epochs=config["surrogate_epochs"],
            learning_rate=config["surrogate_learning_rate"],
            batch_size=config["surrogate_batch_size"],
            seed=config["surrogate_seed"],
            restarts=config["surrogate_restarts"],
            device=self.device,
        )
        self._surrogate = surrogate.eval()
        self._fit_info = info
        return self

    def infer(self, defense_input: DefenseInput) -> DefenseOutput:
        # BaseDefense.run() fits before inferring, so only retrain when a
        # direct infer() call arrives unfitted or carries a different
        # defense_config -- run() must not pay the surrogate training twice.
        # Calibration DATA changes are not detected: callers going through
        # infer() directly should fit() again themselves in that case.
        if (
            self._surrogate is None
            or self._merge_config(defense_input.defense_config) != self._effective_config
        ):
            self.fit(defense_input)
        config = self._effective_config
        probs = self._input_probabilities(defense_input)
        if probs.shape[1] != self._fit_info["num_classes"]:
            raise ValueError(
                "posteriors to protect must match the surrogate's class dimension "
                f"({self._fit_info['num_classes']})."
            )

        start = time.perf_counter()
        protected, stats = perturb_posteriors(
            probs, self._surrogate, **self._solver_config(config)
        )
        self._last_perturb_stats = stats
        self._last_perturb_seconds = time.perf_counter() - start

        if self.defended_model is not None:
            # The predictor snapshots the surrogate and solver parameters, so
            # later refits of this defense never retroactively change released
            # predictors.  No .eval() here: it would recurse into the wrapped
            # user model and flip its training mode permanently; forward()
            # manages the wrapped model's mode itself.
            self.protected_predictor = _MemGuardProtectedPredictor(
                self.defended_model, self._surrogate, self._solver_config(config)
            ).to(self.device)

        return DefenseOutput(
            defended_model=self.defended_model,
            protected_predictor=self.protected_predictor,
            protected_outputs=protected.numpy(),
            artifacts={
                "memguard_config": dict(config),
                "surrogate": dict(self._fit_info),
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
                "surrogate_train_accuracy": self._fit_info.get("train_accuracy"),
                "perturbed_fraction": stats["perturbed_fraction"],
                "argmax_preserved_fraction": stats["argmax_preserved_fraction"],
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
            "surrogate_hidden_dims": list(self.surrogate_hidden_dims),
            "surrogate_epochs": self.surrogate_epochs,
            "surrogate_learning_rate": self.surrogate_learning_rate,
            "surrogate_batch_size": self.surrogate_batch_size,
            "surrogate_seed": self.surrogate_seed,
            "surrogate_restarts": self.surrogate_restarts,
            "c1": self.c1,
            "c2": self.c2,
            "c3_init": self.c3_init,
            "c3_growth": self.c3_growth,
            "c3_max": self.c3_max,
            "step_size": self.step_size,
            "max_iterations": self.max_iterations,
            "crossing_tolerance": self.crossing_tolerance,
            "batch_size": self.batch_size,
        }
        overrides = dict(overrides or {})
        unknown = sorted(set(overrides) - set(config))
        if unknown:
            raise ValueError(
                "unknown defense_config keys for MemGuardDefense (this API has no "
                "entropy-target parameters): " + ", ".join(unknown)
            )
        config.update(overrides)
        config["surrogate_hidden_dims"] = [
            int(width) for width in config["surrogate_hidden_dims"]
        ]
        for key in (
            "surrogate_epochs",
            "surrogate_batch_size",
            "surrogate_seed",
            "surrogate_restarts",
            "max_iterations",
            "batch_size",
        ):
            config[key] = int(config[key])
        for key in (
            "surrogate_learning_rate",
            "c1",
            "c2",
            "c3_init",
            "c3_growth",
            "c3_max",
            "step_size",
            "crossing_tolerance",
        ):
            config[key] = float(config[key])
        if not config["surrogate_hidden_dims"] or any(
            width <= 0 for width in config["surrogate_hidden_dims"]
        ):
            raise ValueError("surrogate_hidden_dims must be non-empty and positive.")
        for key in (
            "surrogate_epochs",
            "surrogate_batch_size",
            "surrogate_restarts",
            "max_iterations",
            "batch_size",
        ):
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive.")
        if config["surrogate_learning_rate"] <= 0.0:
            raise ValueError("surrogate_learning_rate must be positive.")
        if config["c1"] < 0.0 or config["c2"] < 0.0 or config["c3_init"] <= 0.0:
            raise ValueError("c1/c2 must be non-negative and c3_init positive.")
        if config["c3_growth"] <= 1.0:
            raise ValueError("c3_growth must exceed 1.")
        if config["c3_max"] < config["c3_init"]:
            raise ValueError("c3_max must be at least c3_init.")
        if config["step_size"] <= 0.0:
            raise ValueError("step_size must be positive.")
        if config["crossing_tolerance"] < 0.0:
            raise ValueError("crossing_tolerance must be non-negative.")
        return config

    @staticmethod
    def _solver_config(config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: config[key]
            for key in (
                "c1",
                "c2",
                "c3_init",
                "c3_growth",
                "c3_max",
                "step_size",
                "max_iterations",
                "crossing_tolerance",
            )
        }

    def _posteriors_from(
        self, precomputed: Any, sources: Sequence[Any], key: str
    ) -> Optional[torch.Tensor]:
        """Posteriors from `precomputed`, or by running the model on `sources`."""
        if precomputed is not None:
            probs = torch.as_tensor(precomputed, dtype=torch.float32).detach()
            if probs.ndim != 2:
                raise ValueError(f"auxiliary_data[{key!r}] must be 2-D.")
            return probs
        for source in sources:
            if source is not None and self.defended_model is not None:
                logits = predict_logits(
                    self.defended_model,
                    source,
                    device=self.device,
                    batch_size=self._effective_config["batch_size"],
                )
                return torch.softmax(logits, dim=1)
        return None

    def _member_probabilities(self, defense_input: DefenseInput) -> Optional[torch.Tensor]:
        auxiliary = defense_input.auxiliary_data or {}
        return self._posteriors_from(
            auxiliary.get("member_probabilities"),
            (defense_input.train_data,),
            "member_probabilities",
        )

    def _nonmember_probabilities(self, defense_input: DefenseInput) -> Optional[torch.Tensor]:
        auxiliary = defense_input.auxiliary_data or {}
        return self._posteriors_from(
            auxiliary.get("nonmember_probabilities"),
            (auxiliary.get("nonmember_data"), defense_input.test_data),
            "nonmember_probabilities",
        )

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
                "MemGuardDefense requires posteriors to protect: target_model plus "
                "samples, or signals['probabilities'] / signals['logits']."
            )
        if probs.ndim != 2 or probs.shape[1] < 2:
            raise ValueError("MemGuard requires posteriors of shape (batch, classes >= 2).")
        return probs

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
    "entropy_of_probabilities",
    "perturb_posteriors",
    "train_surrogate_attack_model",
    "tpr_at_fpr",
]
