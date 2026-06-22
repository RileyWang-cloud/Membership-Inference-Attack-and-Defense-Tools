# Membership-Inference-Attack-and-Defense-Tools

This project is an open-source research toolkit for studying privacy inference attacks and defense mechanisms in machine learning models. Motivated by the severe risks of membership privacy leakage in ML models, the toolkit integrates existing attack and defense methods to support systematic evaluation and benchmarking of privacy risks.

## Unified interface

Every attack and defense follows the same minimal contract, so methods are interchangeable inside one pipeline. The full specification lives in [ATTACK_INTERFACE_DESIGN_ZH.md](ATTACK_INTERFACE_DESIGN_ZH.md) and [DEFENSE_INTERFACE_DESIGN_ZH.md](DEFENSE_INTERFACE_DESIGN_ZH.md).

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

`run()` = `fit()` → `infer()` → `evaluate()`; `evaluate()` runs only when `membership_labels` is provided. The main result is always `membership_scores` (**higher = more likely member** — each attack flips its raw signal if necessary to honour this).

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

## Attack methods

All classes subclass `BaseAttack` and consume `AttackInput` → produce `AttackOutput`.

| Class | File | Family | Demo |
|---|---|---|---|
| `LossAttack`, `CorrectnessAttack`, `ConfidenceAttack`, `EntropyAttack`, `ModifiedEntropyAttack` | [metric_based.py](Attack/metric_based.py) | signal-only (no training) | [metric_based_demo.py](Attack/metric_based_demo.py) |
| `ShadowBasedAttack` | [shadow_based.py](Attack/shadow_based.py) | shadow-model | [shadow_based_demo.py](Attack/shadow_based_demo.py) |
| `LiRAAttack` | [lira.py](Attack/lira.py) | reference-model likelihood ratio | [lira_demo.py](Attack/lira_demo.py) |
| `RMIAAttack` | [rmia.py](Attack/rmia.py) | reference-model ratio-of-ratios | [rmia_demo.py](Attack/rmia_demo.py) |
| `SecMIAAttack` | [secmia.py](Attack/secmia.py) | shadow-model (SeCMIA) | [secmia_demo.py](Attack/secmia_demo.py) |
| `GSAMIAAttack` | [gsamia.py](Attack/gsamia.py) | gradient-signal features | [gsamia_demo.py](Attack/gsamia_demo.py) |
| `QMIAAttack` | [qmia.py](Attack/qmia.py) | quantile regression on confidence margin | [qmia_demo.py](Attack/qmia_demo.py) |
| `RAPIDAttack` | [rapid.py](Attack/rapid.py) | data-augmentation | [rapid_demo.py](Attack/rapid_demo.py) |
| `EnhancedMIAAttack` | [enhanced_mia.py](Attack/enhanced_mia.py) | learned NN on enhanced per-sample features | [enhanced_mia_demo.py](Attack/enhanced_mia_demo.py) |
| `GANLeaksAttack` | [gan_leaks.py](Attack/gan_leaks.py) | generative-model reconstruction (FBB / PBB) | [gan_leaks_demo.py](Attack/gan_leaks_demo.py) |
| `LOGANAttack` | [logan_attack.py](Attack/logan_attack.py) | discriminator-confidence for generative models | [logan_demo.py](Attack/logan_demo.py) |
| `BiasedMIAAttack` | [biased_mia.py](Attack/biased_mia.py) | classifier on interaction-vs-recommendation vectors | — |
| `DLMIAAttack` | [dl_mia.py](Attack/dl_mia.py) | 3-branch fused MLP (recommender) | — |
| `MEMIAAttack` | [me_mia.py](Attack/me_mia.py) | classifier on per-user score features | — |
| `ShadowFreeMIAAttack` | [shadow_free_mia.py](Attack/shadow_free_mia.py) | embedding similarity, no shadow / no target query | [shadow_free_mia_demo.py](Attack/shadow_free_mia_demo.py) |
| `TransferAttack`, `BoundaryAttack` | [transfer_attack.py](Attack/transfer_attack.py) | transfer & decision-boundary | [transfer_boundary_demo.py](Attack/transfer_boundary_demo.py) |

## Defense methods

All classes subclass `BaseDefense` and consume `DefenseInput` → produce `DefenseOutput`.

| Class | File | Family | Demo |
|---|---|---|---|
| `DPSGDDefense` | [dp_sgd.py](Defense/dp_sgd.py) | DP-SGD for PyTorch classifiers (training-time) | [dp_sgd_demo.py](Defense/dp_sgd_demo.py) |
| `VAEDPDefense` | [vae_dp.py](Defense/vae_dp.py) | DP-SGD-trained VAE against reconstruction MIA | [vae_dp_demo.py](Defense/vae_dp_demo.py) |
| `PopularityRandomizationDefense`, `RecommendationListShuffleDefense` | [rec_privacy_defenses.py](Defense/rec_privacy_defenses.py) | recommender output-processing | [rec_privacy_defense_demo.py](Defense/rec_privacy_defense_demo.py) |

## Quick start

Each `*_demo.py` is a self-contained end-to-end example: it builds a synthetic target model, runs the attack/defense through `run()`, and asserts its self-checks.

```bash
python Attack/qmia_demo.py
python Attack/gan_leaks_demo.py
python Attack/enhanced_mia_demo.py
python Attack/shadow_free_mia_demo.py
python Defense/vae_dp_demo.py
```

Dependencies: PyTorch, scikit-learn, NumPy. The metric-based and shadow-free demos are NumPy-only.

## Repository layout

```text
Attack/        attack implementations + Attack/base.py (interface) + *_demo.py
Defense/       defense implementations + Defense/base.py (interface) + *_demo.py
Ref/           original third-party reference snapshots (read-only, not imported)
ATTACK_INTERFACE_DESIGN_ZH.md   unified attack interface spec
DEFENSE_INTERFACE_DESIGN_ZH.md  unified defense interface spec
```

Implementations under `Ref/` are kept only as paper references; the active code in `Attack/` and `Defense/` is self-contained and conforms to the unified interface above.
