from typing import Optional

import numpy as np
import torch
from torch import Tensor


# TODO: check this code for integration
def sequence_mask(length: Tensor, max_length: Optional[int] = None) -> Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def vpsde_beta_t(t, T, min_beta, max_beta):
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def get_noise_schedule_list(schedule_mode, timesteps, min_beta=0.0, max_beta=0.01, s=0.008):
    if schedule_mode == "linear":
        schedule_list = np.linspace(1e-4, max_beta, timesteps)
    elif schedule_mode == "cosine":
        steps = timesteps + 1
        x = np.linspace(0, steps, steps)
        alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        schedule_list = np.clip(betas, a_min=0, a_max=0.999)
    elif schedule_mode == "vpsde":
        schedule_list = np.array([
            vpsde_beta_t(t, timesteps, min_beta, max_beta) for t in range(1, timesteps + 1)])
    else:
        raise NotImplementedError
    return schedule_list
