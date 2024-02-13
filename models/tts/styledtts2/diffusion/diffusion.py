from math import pi
from typing import Optional, Tuple

from einops import rearrange, reduce
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .distributions import Distribution
from .utils import default, exists

""" Diffusion Classes """


def pad_dims(x: Tensor, ndim: int) -> Tensor:
    r"""Pads additional dimensions to the right of the tensor.

    Args:
        x (Tensor): The input tensor.
        ndim (int): The number of dimensions to add.

    Returns:
        Tensor: The padded tensor.
    """
    return x.view(*x.shape, *((1,) * ndim))


def clip(x: Tensor, dynamic_threshold: float = 0.0):
    r"""Clips the input tensor between -1.0 and 1.0, or between -scale and scale if dynamic_threshold is not 0.0.

    Args:
        x (Tensor): The input tensor.
        dynamic_threshold (float, optional): The dynamic threshold for clipping. Defaults to 0.0.

    Returns:
        Tensor: The clipped tensor.
    """
    if dynamic_threshold == 0.0:
        return x.clamp(-1.0, 1.0)
    else:
        # Dynamic thresholding
        # Find dynamic threshold quantile for each batch
        x_flat = rearrange(x, "b ... -> b (...)")
        scale = torch.quantile(x_flat.abs(), dynamic_threshold, dim=-1)

        # Clamp to a min of 1.0
        scale.clamp_(min=1.0)

        # Clamp all values and scale
        scale = pad_dims(scale, ndim=x.ndim - scale.ndim)
        return x.clamp(-scale, scale) / scale


def to_batch(
    batch_size: int,
    x: Optional[float] = None,
    xs: Optional[Tensor] = None,
) -> Tensor:
    r"""Converts a scalar or a tensor to a batch of tensors.

    Args:
        batch_size (int): The batch size.
        x (Optional[float], optional): The scalar to convert. Defaults to None.
        xs (Optional[Tensor], optional): The tensor to convert. Defaults to None.

    Returns:
        Tensor: The batch of tensors.
    """
    assert exists(x) ^ exists(xs), "Either x or xs must be provided"
    # If x provided use the same for all batch items
    if exists(x):
        xs = torch.full(size=(batch_size,), fill_value=x)
    assert exists(xs)
    return xs


class Diffusion(nn.Module):
    r"""Base class for diffusion models."""

    alias: str = ""

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        r"""Denoises the input tensor.

        Args:
            x_noisy (Tensor): The noisy input tensor.
            sigmas (Optional[Tensor], optional): The noise levels. Defaults to None.
            sigma (Optional[float], optional): The noise level. Defaults to None.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.

        Returns:
            Tensor: The denoised tensor.
        """
        raise NotImplementedError("Diffusion class missing denoise_fn")

    def forward(self, x: Tensor, noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Forward pass of the diffusion model.

        Args:
            x (Tensor): The input tensor.
            noise (Tensor, optional): The noise tensor. Defaults to torch.tensor([]).
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method should be overridden by subclasses.

        Returns:
            Tensor: The output tensor.
        """
        raise NotImplementedError("Diffusion class missing forward function")


class VDiffusion(Diffusion):
    r"""VDiffusion class that extends the base Diffusion class."""

    alias = "v"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        r"""Initialize the VDiffusion with a network and a sigma distribution.

        Args:
            net (nn.Module): The network module.
            sigma_distribution (Distribution): The sigma distribution.
        """
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_alpha_beta(self, sigmas: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Get alpha and beta values based on the input sigmas.

        Args:
            sigmas (Tensor): The input sigmas.

        Returns:
            Tuple[Tensor, Tensor]: The alpha and beta values.
        """
        angle = sigmas * pi / 2
        alpha = torch.cos(angle)
        beta = torch.sin(angle)
        return alpha, beta

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        r"""Denoise the input tensor.

        Args:
            x_noisy (Tensor): The noisy input tensor.
            sigmas (Optional[Tensor], optional): The noise levels. Defaults to None.
            sigma (Optional[float], optional): The noise level. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The denoised tensor.
        """
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(batch_size, x=sigma, xs=sigmas).to(device=device)
        return self.net(x_noisy, sigmas, **kwargs)

    def forward(self, x: Tensor, noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Forward pass of the VDiffusion model.

        Args:
            x (Tensor): The input tensor.
            noise (Tensor, optional): The noise tensor. Defaults to torch.tensor([]).
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output tensor.
        """
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size).to(device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Get noise
        noise = default(noise, torch.randn_like(x)).to(device=device)

        # Combine input and noise weighted by half-circle
        alpha, beta = self.get_alpha_beta(sigmas_padded)
        x_noisy = x * alpha + noise * beta
        x_target = noise * alpha - x * beta

        # Denoise and return loss
        x_denoised = self.denoise_fn(x_noisy, sigmas, **kwargs)
        return F.mse_loss(x_denoised, x_target)


class KDiffusion(Diffusion):
    r"""Elucidated Diffusion (Karras et al. 2022): https://arxiv.org/abs/2206.00364"""

    alias = "k"

    def __init__(
        self,
        net: nn.Module,
        *,
        sigma_distribution: Distribution,
        sigma_data: float,  # data distribution standard deviation
        dynamic_threshold: float = 0.0,
    ):
        r"""Initialize the KDiffusion with a network, a sigma distribution, sigma data, and a dynamic threshold.

        Args:
            net (nn.Module): The network module.
            sigma_distribution (Distribution): The sigma distribution.
            sigma_data (float): The data distribution standard deviation.
            dynamic_threshold (float, optional): The dynamic threshold. Defaults to 0.0.
        """
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data
        self.sigma_distribution = sigma_distribution
        self.dynamic_threshold = dynamic_threshold

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        r"""Get scale weights based on the input sigmas.

        Args:
            sigmas (Tensor): The input sigmas.

        Returns:
            Tuple[Tensor, ...]: The scale weights.
        """
        sigma_data = self.sigma_data

        c_noise = torch.log(sigmas) * 0.25
        sigmas = rearrange(sigmas, "b -> b 1 1")

        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5

        return c_skip, c_out, c_in, c_noise

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        r"""Denoise the input tensor.

        Args:
            x_noisy (Tensor): The noisy input tensor.
            sigmas (Optional[Tensor], optional): The noise levels. Defaults to None.
            sigma (Optional[float], optional): The noise level. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The denoised tensor.
        """
        batch_size, device = x_noisy.shape[0], x_noisy.device

        sigmas = to_batch(batch_size, x=sigma, xs=sigmas).to(device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in, c_noise = self.get_scale_weights(sigmas)

        x_pred = self.net(c_in * x_noisy, c_noise, **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred

        return x_denoised

    def loss_weight(self, sigmas: Tensor) -> Tensor:
        r"""Computes weight depending on data distribution.

        Args:
            sigmas (Tensor): The input sigmas.

        Returns:
            Tensor: The loss weight.
        """
        return (sigmas ** 2 + self.sigma_data ** 2) * (sigmas * self.sigma_data) ** -2

    def forward(self, x: Tensor, noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Forward pass of the KDiffusion model.

        Args:
            x (Tensor): The input tensor.
            noise (Optional[Tensor], optional): The noise tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output tensor.
        """
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size).to(device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, torch.randn_like(x)).to(device=device)
        x_noisy = x + sigmas_padded * noise

        # Compute denoised values
        x_denoised = self.denoise_fn(x_noisy, sigmas=sigmas, **kwargs)

        # Compute weighted loss
        losses = F.mse_loss(x_denoised, x, reduction="none")
        losses = reduce(losses, "b ... -> b", "mean")
        losses = losses * self.loss_weight(sigmas)
        loss = losses.mean()
        return loss


class VKDiffusion(Diffusion):
    r"""VKDiffusion class that extends the base Diffusion class."""

    alias = "vk"

    def __init__(self, net: nn.Module, *, sigma_distribution: Distribution):
        r"""Initialize the VKDiffusion with a network and a sigma distribution.

        Args:
            net (nn.Module): The network module.
            sigma_distribution (Distribution): The sigma distribution.
        """
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution

    def get_scale_weights(self, sigmas: Tensor) -> Tuple[Tensor, ...]:
        r"""Get scale weights based on the input sigmas.

        Args:
            sigmas (Tensor): The input sigmas.

        Returns:
            Tuple[Tensor, ...]: The scale weights.
        """
        sigma_data = 1.0
        sigmas = rearrange(sigmas, "b -> b 1 1")
        c_skip = (sigma_data ** 2) / (sigmas ** 2 + sigma_data ** 2)
        c_out = -sigmas * sigma_data * (sigma_data ** 2 + sigmas ** 2) ** -0.5
        c_in = (sigmas ** 2 + sigma_data ** 2) ** -0.5
        return c_skip, c_out, c_in

    def sigma_to_t(self, sigmas: Tensor) -> Tensor:
        r"""Convert sigmas to t.

        Args:
            sigmas (Tensor): The input sigmas.

        Returns:
            Tensor: The converted t.
        """
        return sigmas.atan() / pi * 2

    def t_to_sigma(self, t: Tensor) -> Tensor:
        r"""Convert t to sigmas.

        Args:
            t (Tensor): The input t.

        Returns:
            Tensor: The converted sigmas.
        """
        return (t * pi / 2).tan()

    def denoise_fn(
        self,
        x_noisy: Tensor,
        sigmas: Optional[Tensor] = None,
        sigma: Optional[float] = None,
        **kwargs,
    ) -> Tensor:
        r"""Denoise the input tensor.

        Args:
            x_noisy (Tensor): The noisy input tensor.
            sigmas (Optional[Tensor], optional): The noise levels. Defaults to None.
            sigma (Optional[float], optional): The noise level. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The denoised tensor.
        """
        batch_size, device = x_noisy.shape[0], x_noisy.device
        sigmas = to_batch(batch_size, x=sigma, xs=sigmas).to(device=device)

        # Predict network output and add skip connection
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)
        x_denoised = c_skip * x_noisy + c_out * x_pred
        return x_denoised

    def forward(self, x: Tensor, noise: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Forward pass of the VKDiffusion model.

        Args:
            x (Tensor): The input tensor.
            noise (Optional[Tensor], optional): The noise tensor. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            Tensor: The output tensor.
        """
        batch_size, device = x.shape[0], x.device

        # Sample amount of noise to add for each batch element
        sigmas = self.sigma_distribution(num_samples=batch_size).to(device=device)
        sigmas_padded = rearrange(sigmas, "b -> b 1 1")

        # Add noise to input
        noise = default(noise, torch.randn_like(x)).to(device=device)
        x_noisy = x + sigmas_padded * noise

        # Compute model output
        c_skip, c_out, c_in = self.get_scale_weights(sigmas)
        x_pred = self.net(c_in * x_noisy, self.sigma_to_t(sigmas), **kwargs)

        # Compute v-objective target
        v_target = (x - c_skip * x_noisy) / (c_out + 1e-7)

        # Compute loss
        loss = F.mse_loss(x_pred, v_target)
        return loss
