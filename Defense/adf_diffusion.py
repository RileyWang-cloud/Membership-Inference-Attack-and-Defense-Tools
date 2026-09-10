"""Selective diffusion building blocks used by the ADF training-time defense."""
from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    return values.gather(0, timesteps).float().view([timesteps.shape[0]] + [1] * (len(shape) - 1))


def generate_t_to_group(total_steps: int, num_t_groups: int) -> list[int]:
    if total_steps < num_t_groups or num_t_groups < 1:
        raise ValueError("Require total_steps >= num_t_groups >= 1.")
    width = total_steps // num_t_groups
    return [min(t // width, num_t_groups - 1) for t in range(total_steps)]


class SelectiveDenoiser(nn.Module):
    """Inference model that routes every sample to its timestep-specific UNet."""
    def __init__(self, models: Sequence[nn.Module], t_to_group: Sequence[int], mask: torch.Tensor | None = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("t_to_group", torch.as_tensor(t_to_group, dtype=torch.long))
        if mask is not None:
            self.register_buffer("M", mask.detach().clone())

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        group_ids = self.t_to_group[timesteps.detach().cpu()].to(x.device)
        output = torch.empty_like(x)
        for group in group_ids.unique():
            indices = (group_ids == group).nonzero(as_tuple=False).flatten()
            output[indices] = self.models[int(group)](x[indices], timesteps[indices])
        return output


class SelectiveGaussianDiffusionTrainer(nn.Module):
    """Masked grouped DDPM objective used by ADF (the SMCD training objective)."""
    def __init__(self, model_factory: Callable[[], nn.Module], beta_1: float, beta_T: float,
                 total_steps: int, k: int, num_t_groups: int, t_to_group: Sequence[int],
                 sparsity: float, mask: torch.Tensor | None = None):
        super().__init__()
        self.T, self.k, self.num_t_groups = total_steps, k, num_t_groups
        self.models = nn.ModuleList([model_factory() for _ in range(num_t_groups)])
        self.register_buffer("t_to_group", torch.as_tensor(t_to_group, dtype=torch.long))
        if mask is None:
            mask = self.generate_mask(k, total_steps, sparsity)
        self.register_buffer("M", mask.float())
        betas = torch.linspace(beta_1, beta_T, total_steps).double()
        alphas_bar = torch.cumprod(1.0 - betas, dim=0)
        self.register_buffer("sqrt_alphas_bar", torch.sqrt(alphas_bar))
        self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1.0 - alphas_bar))

    @staticmethod
    def generate_mask(k: int, total_steps: int, sparsity: float) -> torch.Tensor:
        keep = max(1, min(k, round((1.0 - sparsity) * k)))
        mask = torch.zeros((k, total_steps), dtype=torch.float32)
        for timestep in range(total_steps):
            mask[torch.randperm(k)[:keep], timestep] = 1
        return mask

    def _loss(self, images: torch.Tensor, group_ids: torch.Tensor, timestep: torch.Tensor,
              apply_mask: bool) -> torch.Tensor:
        noise = torch.randn_like(images)
        noisy = extract(self.sqrt_alphas_bar, timestep, images.shape) * images + \
            extract(self.sqrt_one_minus_alphas_bar, timestep, images.shape) * noise
        valid = self.M[group_ids, timestep].bool() if apply_mask else torch.ones_like(timestep, dtype=torch.bool)
        if not valid.any():
            return images.sum() * 0.0
        indices = valid.nonzero(as_tuple=False).flatten()
        output = torch.empty_like(noise[indices])
        routed_groups = self.t_to_group[timestep[indices].cpu()].to(images.device)
        for group in routed_groups.unique():
            routed = (routed_groups == group).nonzero(as_tuple=False).flatten()
            output[routed] = self.models[int(group)](noisy[indices][routed], timestep[indices][routed])
        return F.mse_loss(output, noise[indices])

    def forward(self, images: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(self.T, (images.shape[0],), device=images.device)
        return self._loss(images, group_ids, timesteps, apply_mask=True)

    def loss_at_timestep(self, images: torch.Tensor, group_ids: torch.Tensor, timestep: int) -> torch.Tensor:
        ts = torch.full((images.shape[0],), timestep, dtype=torch.long, device=images.device)
        return self._loss(images, group_ids, ts, apply_mask=False)

    def as_denoiser(self) -> SelectiveDenoiser:
        return SelectiveDenoiser(list(self.models), self.t_to_group.tolist(), self.M)
