"""Generate DDPM samples and evaluate them with the local FID implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from .score.fid import get_fid_score


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, timesteps).float().view([timesteps.shape[0]] + [1] * (len(shape) - 1))


def _sample_ddpm(model: torch.nn.Module, flags_obj: Any, noise: torch.Tensor) -> torch.Tensor:
    """Algorithm 2 sampler, kept here because GSAMIA has no diffusion module."""
    steps = int(getattr(flags_obj, "T", 1000))
    beta_1 = float(getattr(flags_obj, "beta_1", 1e-4))
    beta_t = float(getattr(flags_obj, "beta_T", 0.02))
    var_type = str(getattr(flags_obj, "var_type", "fixedlarge"))
    mean_type = str(getattr(flags_obj, "mean_type", "epsilon"))
    if mean_type != "epsilon":
        raise ValueError("GSAMIA FID sampling currently supports DDPM mean_type='epsilon' only.")
    if var_type not in {"fixedlarge", "fixedsmall"}:
        raise ValueError(f"Unsupported DDPM var_type for FID: {var_type}")

    betas = torch.linspace(beta_1, beta_t, steps, device=noise.device, dtype=torch.float64)
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = torch.nn.functional.pad(alphas_bar, [1, 0], value=1)[:steps]
    posterior_var = betas * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    posterior_log_var = torch.log(torch.cat([posterior_var[1:2], posterior_var[1:]]))
    log_variances = (
        torch.log(torch.cat([posterior_var[1:2], betas[1:]]))
        if var_type == "fixedlarge" else posterior_log_var
    )
    coef1 = torch.sqrt(alphas_bar_prev) * betas / (1.0 - alphas_bar)
    coef2 = torch.sqrt(alphas) * (1.0 - alphas_bar_prev) / (1.0 - alphas_bar)
    reciprocal_alphas_bar = torch.sqrt(1.0 / alphas_bar)
    reciprocal_minus_one = torch.sqrt(1.0 / alphas_bar - 1.0)

    sample = noise
    for time_step in reversed(range(steps)):
        timestep = torch.full((noise.shape[0],), time_step, dtype=torch.long, device=noise.device)
        predicted_noise = model(sample, timestep)
        x0 = (_extract(reciprocal_alphas_bar, timestep, sample.shape) * sample -
              _extract(reciprocal_minus_one, timestep, sample.shape) * predicted_noise).clamp(-1, 1)
        mean = (_extract(coef1, timestep, sample.shape) * x0 +
                _extract(coef2, timestep, sample.shape) * sample)
        if time_step > 0:
            sample = mean + torch.exp(0.5 * _extract(log_variances, timestep, sample.shape)) * torch.randn_like(sample)
        else:
            sample = mean
    return sample.clamp(-1, 1)


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
    """Return FID of DDPM samples against a precomputed dataset statistics cache."""
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

    was_training = model.training
    model.eval()
    images = []
    image_size = int(getattr(flags_obj, "img_size", 32))
    with torch.no_grad():
        for start in range(0, num_images, batch_size):
            current_batch = min(batch_size, num_images - start)
            noise = torch.randn(current_batch, 3, image_size, image_size, device=runtime_device)
            images.append(((_sample_ddpm(model, flags_obj, noise) + 1) / 2).cpu().numpy())
    if was_training:
        model.train()
    return float(get_fid_score(str(cache_path), np.concatenate(images), num_images=num_images,
                               batch_size=batch_size, use_torch=use_torch, verbose=verbose))
