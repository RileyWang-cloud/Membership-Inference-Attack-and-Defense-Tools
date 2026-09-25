"""CIFAR-10 tensors for the benchmark (plan v2 §4.2, W2).

Loads via torchvision (auto-download), applies ToTensor + channel
normalization only (no augmentation anywhere in the benchmark), and exposes
full train/test tensors plus the global-index view used by the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torchvision import datasets, transforms

from benchmark.bmk import common


def _data_root() -> Path:
    root = Path(common.BENCH_ROOT) / "data"
    root.mkdir(parents=True, exist_ok=True)
    return root


def load_cifar10_tensors() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (train_X, train_y, test_X, test_y) normalized float32 tensors.

    Stacks are cached on disk (benchmark/data/cifar_tensors.pt) — every
    AttackContext / training script reloads this; re-stacking 60k images
    through the torchvision Dataset __getitem__ costs ~a minute each time.
    """
    cache = _data_root() / "cifar_tensors.pt"
    if cache.exists():
        blob = torch.load(cache, weights_only=False)
        return blob["train_X"], blob["train_y"], blob["test_X"], blob["test_y"]

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(common.CIFAR_MEAN, common.CIFAR_STD),
        ]
    )
    train_set = datasets.CIFAR10(
        root=str(_data_root()), train=True, download=True, transform=transform
    )
    test_set = datasets.CIFAR10(
        root=str(_data_root()), train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader  # batched stacking, far faster than per-item
    train_X = torch.cat([b for b, _ in loader(train_set, batch_size=4096, num_workers=0)])
    test_X = torch.cat([b for b, _ in loader(test_set, batch_size=4096, num_workers=0)])
    train_y = torch.as_tensor(train_set.targets, dtype=torch.long)
    test_y = torch.as_tensor(test_set.targets, dtype=torch.long)
    assert train_X.shape[0] == common.TRAIN_SIZE and test_X.shape[0] == common.TEST_SIZE
    torch.save({"train_X": train_X, "train_y": train_y, "test_X": test_X, "test_y": test_y}, cache)
    return train_X, train_y, test_X, test_y


def gather_global(indices: np.ndarray, tensors=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Materialize (X, y) for an array of manifest global indices."""
    if tensors is None:
        tensors = load_cifar10_tensors()
    train_X, train_y, test_X, test_y = tensors
    xs, ys = [], []
    for index in np.asarray(indices, dtype=np.int64):
        source, pos = common.global_to_position(int(index))
        if source == "train":
            xs.append(train_X[pos])
            ys.append(train_y[pos])
        else:
            xs.append(test_X[pos])
            ys.append(test_y[pos])
    return torch.stack(xs), torch.as_tensor(ys, dtype=torch.long)
