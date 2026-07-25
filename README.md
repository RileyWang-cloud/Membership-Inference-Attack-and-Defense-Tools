# Membership-Inference-Attack-and-Defense-Tools

This project is an open-source research toolkit for studying privacy inference
attacks and defense mechanisms in machine learning models. Motivated by the
severe risks of membership privacy leakage in ML models, the toolkit integrates
existing attack and defense methods to support systematic evaluation and
benchmarking of privacy risks.

## Unified Interface

Every attack and defense follows the same minimal contract, so methods are
interchangeable inside one pipeline. The full specification lives in
[ATTACK_INTERFACE_DESIGN_ZH.md](ATTACK_INTERFACE_DESIGN_ZH.md) and
[DEFENSE_INTERFACE_DESIGN_ZH.md](DEFENSE_INTERFACE_DESIGN_ZH.md).

**Attacks** ([Attack/base.py](Attack/base.py)):

```python
from Attack.base import AttackInput
from Attack.qmia import QMIAAttack

output = QMIAAttack().run(AttackInput(
    target_model=model,
    samples=X_query,
    labels=y_query,
    shadow_data={"fit_X": X_offline, "fit_y": y_offline},
    membership_labels=membership,   # optional; enables evaluation
))
# output.membership_scores : higher = more likely member
# output.evaluation        : accuracy / AUROC / TPR@low-FPR (when labels given)
```

`run()` = `fit()` -> `infer()` -> `evaluate()`; `evaluate()` runs only when
`membership_labels` is provided. The main result is always
`membership_scores` (higher = more likely member). Each attack flips its raw
signal if necessary to honor this convention.

**Defenses** ([Defense/base.py](Defense/base.py)):

```python
from Defense.base import DefenseInput
from Defense.vae_dp import VAEDPDefense

output = VAEDPDefense().run(DefenseInput(
    train_data=members,
    test_data=nonmembers,
    defense_config={"use_dp": True, "kl_weight": 0.1},
    eval_config={"enabled": True},   # optional; enables evaluation
))
# output.defended_model : the trained, defended model
# output.evaluation     : utility / privacy / efficiency metrics
```

## Attack Methods

All classes subclass `BaseAttack` and consume `AttackInput` to produce
`AttackOutput`.

| Class | File | Family | Demo |
|---|---|---|---|
| `LossAttack`, `CorrectnessAttack`, `ConfidenceAttack`, `EntropyAttack`, `ModifiedEntropyAttack` | [metric_based.py](Attack/metric_based.py) | signal-only, no training | [metric_based_demo.py](Attack/metric_based_demo.py) |
| `ShadowBasedAttack` | [shadow_based.py](Attack/shadow_based.py) | shadow-model | [shadow_based_demo.py](Attack/shadow_based_demo.py) |
| `LiRAAttack` | [lira.py](Attack/lira.py) | reference-model likelihood ratio | [lira_demo.py](Attack/lira_demo.py) |
| `RMIAAttack` | [rmia.py](Attack/rmia.py) | reference-model ratio-of-ratios | [rmia_demo.py](Attack/rmia_demo.py) |
| `SecMIAAttack` | [secmia.py](Attack/secmia.py) | shadow-model, SeCMIA | [secmia_demo.py](Attack/secmia_demo.py) |
| `GSAMIAAttack` | [gsamia.py](Attack/gsamia.py) | gradient-signal features | [gsamia_demo.py](Attack/gsamia_demo.py) |
| `QMIAAttack` | [qmia.py](Attack/qmia.py) | quantile regression on confidence margin | [qmia_demo.py](Attack/qmia_demo.py) |
| `RAPIDAttack` | [rapid.py](Attack/rapid.py) | data-augmentation | [rapid_demo.py](Attack/rapid_demo.py) |
| `EnhancedMIAAttack` | [enhanced_mia.py](Attack/enhanced_mia.py) | learned NN on enhanced per-sample features | [enhanced_mia_demo.py](Attack/enhanced_mia_demo.py) |
| `GANLeaksAttack` | [gan_leaks.py](Attack/gan_leaks.py) | generative-model reconstruction, FBB / PBB | [gan_leaks_demo.py](Attack/gan_leaks_demo.py) |
| `LOGANAttack` | [logan_attack.py](Attack/logan_attack.py) | discriminator-confidence for generative models | [logan_demo.py](Attack/logan_demo.py) |
| `BiasedMIAAttack` | [biased_mia.py](Attack/biased_mia.py) | classifier on interaction-vs-recommendation vectors | [rec_mia_demo.py](Attack/rec_mia_demo.py) |
| `DLMIAAttack` | [dl_mia.py](Attack/dl_mia.py) | 3-branch fused MLP, recommender | [rec_mia_demo.py](Attack/rec_mia_demo.py) |
| `MEMIAAttack` | [me_mia.py](Attack/me_mia.py) | classifier on per-user score features | [rec_mia_demo.py](Attack/rec_mia_demo.py) |
| `CompareMIAAttack` | [compare_mia.py](Attack/compare_mia.py) | target-surrogate NDCG discrepancy features, recommender | [rec_mia_demo.py](Attack/rec_mia_demo.py) |
| `ShadowFreeMIAAttack` | [shadow_free_mia.py](Attack/shadow_free_mia.py) | embedding similarity, no shadow / no target query | [shadow_free_mia_demo.py](Attack/shadow_free_mia_demo.py) |
| `TransferAttack`, `BoundaryAttack` | [transfer_attack.py](Attack/transfer_attack.py) | transfer and decision-boundary | [transfer_boundary_demo.py](Attack/transfer_boundary_demo.py) |

## Recommender Summary

For recommender-system membership inference, the current unified attack set is:

- `ME-MIA`
- `Biased-MIA`
- `DL-MIA`
- `COMPARE` (proposed method)

Run the unified recommender attack demo:

```bash
python Attack/rec_mia_demo.py
python Attack/rec_mia_demo.py --method compare
```

The recommender defense set currently includes two mechanisms:

- `PopularityRandomizationDefense`
- `RecommendationListShuffleDefense`

## Defense Methods

All classes subclass `BaseDefense` and consume `DefenseInput` to produce
`DefenseOutput`.

| Class | File | Family | Demo |
|---|---|---|---|
| `DPSGDDefense` | [dp_sgd.py](Defense/dp_sgd.py) | DP-SGD for PyTorch classifiers, training-time | [dp_sgd_demo.py](Defense/dp_sgd_demo.py) |
| `VAEDPDefense` | [vae_dp.py](Defense/vae_dp.py) | DP-SGD-trained VAE against reconstruction MIA | [vae_dp_demo.py](Defense/vae_dp_demo.py) |
| `RelaxLossDefense` | [relax_loss.py](Defense/relax_loss.py) | alternating loss relaxation and posterior flattening, training-time | [relax_loss_demo.py](Defense/relax_loss_demo.py) |
| `HAMPDefense` | [hamp.py](Defense/hamp.py) | high-entropy training and rank-preserving output modification, hybrid | [hamp_demo.py](Defense/hamp_demo.py) |
| `EarlyStopDefense` | [early_stop.py](Defense/early_stop.py) | validation-monitored or fixed-epoch early stopping, training-time | [early_stop_demo.py](Defense/early_stop_demo.py) |
| `AdvRegDefense` | [adv_reg.py](Defense/adv_reg.py) | adversarial membership regularization, training-time | [adv_reg_demo.py](Defense/adv_reg_demo.py) |
| `PopularityRandomizationDefense`, `RecommendationListShuffleDefense` | [rec_privacy_defenses.py](Defense/rec_privacy_defenses.py) | recommender output-processing | [rec_privacy_defense_demo.py](Defense/rec_privacy_defense_demo.py) |

The four classifier defenses above accept PyTorch models returning a logits
tensor, a tuple whose first item is logits, or a mapping with a `logits` item.
`AdvRegDefense` additionally requires a disjoint pseudo-non-member set in
`auxiliary_data`; `HAMPDefense` accepts a dataset-specific random non-member
generator there. `EarlyStopDefense` uses validation monitoring by default and
supports the fixed-checkpoint protocol from the reference evaluation through
`defense_config={"stop_epoch": ...}`.

## Quick Start

Each `*_demo.py` is a self-contained end-to-end example: it builds a synthetic
target model, runs the attack/defense through `run()`, and asserts its
self-checks.

```bash
python Attack/qmia_demo.py
python Attack/gan_leaks_demo.py
python Attack/enhanced_mia_demo.py
python Attack/shadow_free_mia_demo.py
python Attack/rec_mia_demo.py --method compare
python Defense/vae_dp_demo.py
python Defense/relax_loss_demo.py
python Defense/hamp_demo.py
python Defense/early_stop_demo.py
python Defense/adv_reg_demo.py
```

Dependencies: PyTorch, scikit-learn, NumPy. The metric-based and shadow-free
demos are NumPy-only.

## Repository Layout

```text
Attack/        attack implementations + Attack/base.py (interface) + *_demo.py
Defense/       defense implementations + Defense/base.py (interface) + *_demo.py
Ref/           original third-party reference snapshots (read-only, not imported)
ATTACK_INTERFACE_DESIGN_ZH.md   unified attack interface spec
DEFENSE_INTERFACE_DESIGN_ZH.md  unified defense interface spec
```

Implementations under `Ref/` are kept only as paper references; the active code
in `Attack/` and `Defense/` is self-contained and conforms to the unified
interface above.
