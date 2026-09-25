"""Model definitions for the CIFAR-10 benchmark (plan v2 §5.3, W3-4090).

- ``resnet18_cifar_factory``: adopts ``Attack/utils_secmia/mia_evals/resnet.py``
  (3x3 stem, no maxpool) as the single ResNet18 definition used by target /
  shadow / reference models (plan §5.3).
- ``resnet18_cifar_groupnorm_factory``: GroupNorm variant replacing every
  BatchNorm2d — required for the DP-SGD defense (plan §10.4), deviation is
  recorded in the defense results.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from benchmark.bmk import common

_RESNET_PATH = common.REPO_ROOT / "Attack" / "utils_secmia" / "mia_evals" / "resnet.py"


def _load_repo_resnet_module():
    spec = importlib.util.spec_from_file_location("repo_mia_evals_resnet", _RESNET_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("repo_mia_evals_resnet", module)
    spec.loader.exec_module(module)
    return module


_repo_resnet = _load_repo_resnet_module()


def resnet18_cifar_factory(num_classes: int = common.NUM_CLASSES) -> nn.Module:
    """ResNet18 with CIFAR-style 3x3 stem — repo's single adopted definition."""
    return _repo_resnet.ResNet18(num_classes=num_classes)


def _to_groupnorm(module: nn.Module, num_groups: int = 8) -> None:
    """Replace every BatchNorm2d with GroupNorm in-place (DP-SGD variant)."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(num_groups, child.num_features))
        else:
            _to_groupnorm(child, num_groups=num_groups)


def resnet18_cifar_groupnorm_factory(num_classes: int = common.NUM_CLASSES) -> nn.Module:
    model = resnet18_cifar_factory(num_classes=num_classes)
    _to_groupnorm(model)
    return model


MODEL_REGISTRY: dict[str, Callable[[], nn.Module]] = {
    "resnet18_cifar": resnet18_cifar_factory,
    "resnet18_cifar_groupnorm": resnet18_cifar_groupnorm_factory,
}


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
