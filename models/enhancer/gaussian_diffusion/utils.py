from inspect import isfunction
from typing import Any, List, Tuple

import numpy as np
import torch
from torch import Size, Tensor


def exists(x: Any) -> bool:
    r"""Check if the input variable is not None.

    Args:
        x (Any): Input variable.

    Returns:
        bool: True if the input variable is not None, False otherwise.
    """
    return x is not None


def default(val: Any, d: Any) -> Any:
    r"""Return the input value if it exists, otherwise return a default value.

    Args:
        val (Any): Input value.
        d (Any): Default value or function to generate the default value.

    Returns:
        Any: Input value if it exists, otherwise the default value.
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(
    a: Tensor,
    t: Tensor,
    x_shape: Size,
):
    r"""Extract elements from tensor 'a' using indices 't'.

    Args:
        a (torch.Tensor): Input tensor.
        t (torch.Tensor): Indices tensor.
        x_shape (Size): Shape of the input tensor 'a'.

    Returns:
        torch.Tensor: Extracted elements tensor.
    """
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(
    shape: Size,
    device: torch.device,
    repeat: bool = False,
):
    r"""Generate random noise tensor with the given shape.

    Args:
        shape (Size): Shape of the noise tensor.
        device (torch.device): Device for the tensor.
        repeat (bool, optional): If True, repeat the noise tensor to match the given shape. Defaults to False.

    Returns:
        torch.Tensor: Random noise tensor.
    """
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def vpsde_beta_t(
    t: int,
    T: int,
    min_beta: float,
    max_beta: float,
):
    r"""Calculate beta coefficient for VPSDE noise schedule at time step t.

    Args:
        t (int): Current time step.
        T (int): Total number of time steps.
        min_beta (float): Minimum value of beta.
        max_beta (float): Maximum value of beta.

    Returns:
        float: Beta coefficient at time step t.
    """
    t_coef = (2 * t - 1) / (T ** 2)
    return 1. - np.exp(-min_beta / T - 0.5 * (max_beta - min_beta) * t_coef)


def get_noise_schedule_list(
    schedule_mode: str,
    timesteps: int,
    min_beta: float = 0.0,
    max_beta: float = 0.01,
    s: float = 0.008,
) -> np.ndarray:
    r"""Generate a noise schedule list based on the specified mode.

    Args:
        schedule_mode (str): Mode for generating the noise schedule. 
                             Can be one of ["linear", "cosine", "vpsde"].
        timesteps (int): Total number of time steps.
        min_beta (float, optional): Minimum value of beta for VPSDE mode. Defaults to 0.0.
        max_beta (float, optional): Maximum value of beta for VPSDE mode. Defaults to 0.01.
        s (float, optional): Parameter for cosine schedule mode. Defaults to 0.008.

    Returns:
        np.ndarray: List or array of beta coefficients for each time step.
    """
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
