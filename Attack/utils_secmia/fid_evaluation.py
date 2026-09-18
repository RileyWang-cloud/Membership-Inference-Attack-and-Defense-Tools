"""Generate DDPM samples and evaluate them with the local FID implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .diffusion import GaussianDiffusionSampler
from .score.fid import get_fid_score


def calculate_model_fid(
    model: torch.nn.Module,
    flags_obj: Any,
    *,
    num_images: int = 100,
    batch_size: Optional[int] = None,
    fid_cache: Optional[str] = None,
    use_torch: bool = False,
    verbose: bool = True,
    device: Optional[str] = None,
) -> float:
    """Return FID of samples from a DDPM model against a precomputed dataset cache.

    This intentionally evaluates generation quality, not membership inference
    quality.  ``num_images=100`` is useful for a smoke test; use 10,000 or more
    images for a reportable CIFAR-10 FID value.
    """
    if num_images < 2:
        raise ValueError("FID requires at least two generated images.")

    runtime_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    batch_size = int(batch_size or getattr(flags_obj, "batch_size", 128))
    if batch_size < 1:
        raise ValueError("FID batch_size must be positive.")

    default_cache = Path(__file__).resolve().parent / "stats" / "cifar10.train.npz"
    cache_path = Path(fid_cache or getattr(flags_obj, "fid_cache", "") or default_cache)
    if not cache_path.is_absolute():
        cache_path = Path(__file__).resolve().parent / cache_path
    if not cache_path.exists():
        raise FileNotFoundError(f"FID statistics cache was not found: {cache_path}")

    sampler = GaussianDiffusionSampler(
        model,
        beta_1=float(getattr(flags_obj, "beta_1", 1e-4)),
        beta_T=float(getattr(flags_obj, "beta_T", 0.02)),
        T=int(getattr(flags_obj, "T", 1000)),
        img_size=int(getattr(flags_obj, "img_size", 32)),
        mean_type=str(getattr(flags_obj, "mean_type", "epsilon")),
        var_type=str(getattr(flags_obj, "var_type", "fixedlarge")),
    ).to(runtime_device)

    was_training = model.training
    model.eval()
    images = []
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            current_batch = min(batch_size, num_images - start)
            noise = torch.randn(
                current_batch,
                3,
                int(getattr(flags_obj, "img_size", 32)),
                int(getattr(flags_obj, "img_size", 32)),
                device=runtime_device,
            )
            images.append(((sampler(noise).clamp(-1, 1) + 1) / 2).cpu().numpy())
    if was_training:
        model.train()

    generated_images = np.concatenate(images, axis=0)
    return float(
        get_fid_score(
            str(cache_path),
            generated_images,
            num_images=num_images,
            batch_size=batch_size,
            use_torch=use_torch,
            verbose=verbose,
        )
    )
