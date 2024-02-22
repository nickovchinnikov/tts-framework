from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from .grad_logp_estimator2d import GradLogPEstimator2d


def get_noise(t: Tensor, beta_init: float, beta_term: float, cumulative: bool = False) -> Tensor:
    r"""Compute noise given time steps.

    Args:
        t (torch.Tensor): Time steps.
        beta_init (float): Initial beta value.
        beta_term (float): Terminal beta value.
        cumulative (bool, optional): Whether to compute cumulative noise. Defaults to False.

    Returns:
        torch.Tensor: Computed noise.
    """
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class Diffusion(Module):
    def __init__(
        self,
        n_feats: int,
        dim: int,
        n_speakers: int = 1,
        spk_emb_dim: int = 64,
        beta_min: float = 0.05,
        beta_max: float = 20.,
        pe_scale: int = 1000,
    ):
        r"""Diffusion block.

        Args:
            n_feats (int): Number of features.
            dim (int): Dimension.
            n_speakers (int, optional): Number of speakers. Defaults to 1.
            spk_emb_dim (int, optional): Speaker embedding dimension. Defaults to 64.
            beta_min (float, optional): Minimum beta value. Defaults to 0.05.
            beta_max (float, optional): Maximum beta value. Defaults to 20.
            pe_scale (int, optional): Positional encoding scale. Defaults to 1000.
        """
        super().__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_speakers = n_speakers
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2d(
            dim,
            n_speakers=n_speakers,
            spk_emb_dim=spk_emb_dim,
            pe_scale=pe_scale,
        )

    def forward_diffusion(self, x0: Tensor, mask: Tensor, mu: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Perform forward diffusion.

        Args:
            x0 (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mu tensor.
            t (torch.Tensor): Time tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Output tensor and noise tensor.
        """
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)

        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)

        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(
        self,
        z: Tensor,
        mask: Tensor,
        mu: Tensor,
        n_timesteps: int,
        stoc: bool = False,
        spk: Optional[torch.Tensor] = None,
    ) -> Tensor:
        r"""Perform reverse diffusion.

        Args:
            z (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mu tensor.
            n_timesteps (int): Number of time steps.
            stoc (bool, optional): Whether to include stochastic term. Defaults to False.
            spk (Optional[torch.Tensor], optional): Speaker tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5)*h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                 device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(
        self,
        z: Tensor,
        mask: Tensor,
        mu: Tensor,
        n_timesteps: int = 50,
        stoc: bool = False,
        spk: Optional[torch.Tensor] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mu tensor.
            n_timesteps (int): Number of time steps.
            stoc (bool, optional): Whether to include stochastic term. Defaults to False.
            spk (Optional[torch.Tensor], optional): Speaker tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(
        self,
        x0: Tensor,
        mask: Tensor,
        mu: Tensor,
        t: Tensor,
        spk: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Compute loss.

        Args:
            x0 (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mu tensor.
            t (torch.Tensor): Time tensor.
            spk (Optional[torch.Tensor], optional): Speaker tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss value and output tensor.
        """
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))

        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(
        self,
        x0: Tensor,
        mask: Tensor,
        mu: Tensor,
        spk: Optional[torch.Tensor] = None,
        offset: float = 1e-5,
    ):
        r"""Compute loss.

        Args:
            x0 (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.
            mu (torch.Tensor): Mu tensor.
            spk (Optional[torch.Tensor], optional): Speaker tensor. Defaults to None.
            offset (float, optional): Offset value. Defaults to 1e-5.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Loss value and output tensor.
        """
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)

        return self.loss_t(x0, mask, mu, t, spk)
