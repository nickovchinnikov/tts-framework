from functools import partial
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from tqdm import tqdm

from models.config.configs import DiffusionConfig

from .denoiser import Denoiser
from .utils import default, extract, get_noise_schedule_list, noise_like


class GaussianDiffusion(nn.Module):
    r"""Class implementing Gaussian Diffusion, a method used for modeling noise in data.

    Diffusion models typically consist of two Markov chains: the diffusion process and the reverse process (or denoising process).
    The diffusion process gradually introduces small Gaussian noises into the data until its structure is completely degraded at step T,
    while the reverse process learns a denoising function to remove the added noise and restore the original data structure.

    In this implementation:
    - The diffusion process models the gradual transformation from the original data `x_0` to the latent variable `x_T` using a predefined variance schedule.
    - The reverse process, implemented as a denoising function, aims to remove the added noise and reconstruct the original data.

    Attributes:
        model (str): Type of diffusion model.
        denoise_fn (Denoiser): Denoising function used in the reverse process.
        mel_bins (int): Number of Mel bins in the data.
        num_timesteps (int): Number of diffusion steps.
        loss_type (str): Type of noise loss used in training.
        betas (Tensor): Variance schedule for the diffusion process.
        alphas_cumprod (Tensor): Cumulative product of (1 - betas).
        alphas_cumprod_prev (Tensor): Cumulative product of (1 - betas) excluding the last element.
        sqrt_alphas_cumprod (Tensor): Square root of alphas_cumprod.
        sqrt_one_minus_alphas_cumprod (Tensor): Square root of (1 - alphas_cumprod).
        log_one_minus_alphas_cumprod (Tensor): Logarithm of (1 - alphas_cumprod).
        sqrt_recip_alphas_cumprod (Tensor): Square root of the reciprocal of alphas_cumprod.
        sqrt_recipm1_alphas_cumprod (Tensor): Square root of the reciprocal of (alphas_cumprod - 1).
        posterior_variance (Tensor): Variance of the posterior distribution.
        posterior_log_variance_clipped (Tensor): Clipped logarithm of the posterior variance.
        posterior_mean_coef1 (Tensor): Coefficient for calculating posterior mean.
        posterior_mean_coef2 (Tensor): Coefficient for calculating posterior mean.
    """

    def __init__(
        self,
        model_config: DiffusionConfig,
    ):
        r"""Initialize the Gaussian Diffusion module.

        Args:
            model_config (DiffusionConfig): Model configuration.
        """
        super().__init__()

        # Model configuration
        self.model = model_config.model
        self.denoise_fn = Denoiser(model_config)

        self.mel_bins = model_config.n_mel_channels

        betas = get_noise_schedule_list(
            schedule_mode=model_config.noise_schedule_naive,
            timesteps=model_config.timesteps if self.model == "naive" else model_config.shallow_timesteps,
            min_beta=model_config.min_beta,
            max_beta=model_config.max_beta,
            s=model_config.s,
        )

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = model_config.noise_loss

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer("posterior_log_variance_clipped", to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer("posterior_mean_coef1", to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer("posterior_mean_coef2", to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Calculate mean, variance, and log variance of the diffusion process.

        Args:
            x_start (Tensor): Input tensor.
            t (Tensor): Time step.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Mean, variance, and log variance.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise: Tensor) -> Tensor:
        r"""Predict the start from noise at a given time step.

        Args:
            x_t (Tensor): Input tensor.
            t (Tensor): Time step.
            noise (Tensor): Noise tensor.

        Returns:
            Tensor: Predicted start from noise.
        """
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Calculate posterior mean, variance, and clipped log variance.

        Args:
            x_start (Tensor): Start tensor.
            x_t (Tensor): Tensor at time step.
            t (Tensor): Time step.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Posterior mean, variance, and clipped log variance.
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def q_posterior_sample(
        self,
        x_start: Tensor,
        x_t: Tensor,
        t: Tensor,
        repeat_noise: bool = False,
    ) -> Tensor:
        r"""Sample from the posterior distribution.

        Args:
            x_start (Tensor): Start tensor.
            x_t (Tensor): Tensor at time step.
            t (Tensor): Time step.
            repeat_noise (bool, optional): Whether to repeat noise. Defaults to False.

        Returns:
            Tensor: Sampled tensor from posterior distribution.
        """
        b, *_, device = *x_start.shape, x_start.device
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=x_t, t=t)
        noise = noise_like(x_start.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x_start.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample(
        self,
        x_t: Tensor,
        t: Tensor,
        cond: Tensor,
        spk_emb: Tensor,
        clip_denoised: bool = True,
    ):
        r"""Sample from the distribution.

        Args:
            x_t (Tensor): Tensor at time step.
            t (Tensor): Time tensor.
            cond (Tensor): Conditional tensor.
            spk_emb (Tensor): Speaker embedding tensor.
            clip_denoised (bool, optional): Whether to clip denoised tensor. Defaults to True.

        Returns:
            Tensor: Sampled tensor.
        """
        b, *_, device = *x_t.shape, x_t.device
        x_0_pred = self.denoise_fn.forward(x_t, t, cond, spk_emb)

        if clip_denoised:
            x_0_pred.clamp_(-1., 1.)

        return self.q_posterior_sample(x_start=x_0_pred, x_t=x_t, t=t)

    @torch.no_grad()
    def interpolate(
        self,
        x1: Tensor,
        x2: Tensor,
        t: int,
        cond: Tensor,
        spk_emb: Tensor,
        lam: float = 0.5,
    ):
        r"""Interpolate between two tensors.

        Args:
            x1 (Tensor): First tensor.
            x2 (Tensor): Second tensor.
            t (int): Time step.
            cond (Tensor): Conditional tensor.
            spk_emb (Tensor): Speaker embedding tensor.
            lam (float, optional): Lambda value. Defaults to 0.5.

        Returns:
            Tensor: Interpolated tensor.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(t)), desc="interpolation sample time step", total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond, spk_emb)

        x = x[:, 0].transpose(1, 2)
        return x
        # return self.denorm_spec(x)

    def q_sample(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Sample from the diffusion process.

        Args:
            x_start (Tensor): Start tensor.
            t (Tensor): Time tensor.
            noise (Tensor, optional): Noise tensor. Defaults to None.

        Returns:
            Tensor: Sampled tensor.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @torch.no_grad()
    def sampling(self, noise: Optional[Tensor] = None) -> List[Tensor]:
        r"""Perform sampling.

        Args:
            noise (Tensor, optional): Noise tensor. Defaults to None.

        Returns:
            List[Tensor]: List of sampled tensors.
        """
        b, *_, device = *self.cond.shape, self.cond.device
        t = self.num_timesteps
        shape = (self.cond.shape[0], 1, self.mel_bins, self.cond.shape[2])
        xs = [torch.randn(shape, device=device) if noise is None else noise]
        for i in tqdm(reversed(range(t)), desc="sample time step", total=t):
            x = self.p_sample(
                xs[-1],
                torch.full((b,), i, device=device, dtype=torch.long),
                self.cond,
                self.spk_emb,
            )
            xs.append(x)
        # output = [self.denorm_spec(x[:, 0].transpose(1, 2)) for x in xs]
        output = [x[:, 0].transpose(1, 2) for x in xs]
        return output

    def diffuse_trace(self, x_start: Tensor, mask: Tensor) -> List[Tensor]:
        r"""Diffuse trace.

        Args:
            x_start (Tensor): Start tensor.
            mask (Tensor): Mask tensor.

        Returns:
            List[Tensor]: List of diffused tensors.
        """
        b, *_, device = *x_start.shape, x_start.device

        # trace = [self.norm_spec(x_start).clamp_(-1., 1.) * ~mask.unsqueeze(-1)]

        trace = [x_start.clamp_(-1., 1.) * ~mask.unsqueeze(-1)]
        for t in range(self.num_timesteps):
            t = torch.full((b,), t, device=device, dtype=torch.long)
            trace.append(
                self.diffuse_fn(x_start, t)[:, 0].transpose(1, 2) * ~mask.unsqueeze(-1),
            )
        return trace

    def diffuse_fn(
        self,
        x_start: Tensor,
        t: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Diffuse function.

        Args:
            x_start (Tensor): Start tensor.
            t (Tensor): Time tensor.
            noise (Tensor, optional): Noise tensor. Defaults to None.

        Returns:
            Tensor: Diffused tensor.
        """
        # x_start = self.norm_spec(x_start)
        # x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
        x_start = x_start[:, None, :, :] # [B, 1, T, M]
        zero_idx = t < 0 # for items where t is -1
        t[zero_idx] = 0
        noise = default(noise, lambda: torch.randn_like(x_start))
        out = self.q_sample(x_start=x_start, t=t, noise=noise)
        out[zero_idx] = x_start[zero_idx] # set x_{-1} as the gt mel
        return out

    def forward(
        self,
        mel: Tensor,
        cond: Tensor,
        spk_emb: Tensor,
        mel_mask: Tensor,
        coarse_mel: Tensor,
        clip_denoised: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Forward pass.

        Args:
            mel (Tensor, optional): Mel tensor. Defaults to None.
            cond (Tensor): Conditional tensor.
            spk_emb (Tensor): Speaker embedding tensor.
            mel_mask (Tensor): Mel mask tensor.
            coarse_mel (Tensor, optional): Coarse mel tensor. Defaults to None.
            clip_denoised (bool, optional): Whether to clip denoised tensor. Defaults to True.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: Tuple containing predicted start, tensor at time step, tensor at previous time step, predicted tensor at previous time step, and time tensor.
        """
        b, *_, device = *cond.shape, cond.device

        # x_t = x_t_prev = x_t_prev_pred = t = None

        mel_mask = ~mel_mask.unsqueeze(-1)
        cond = cond.transpose(1, 2)
        self.cond = cond.detach()
        self.spk_emb = spk_emb.detach()

        if mel is None:
            if self.model != "shallow":
                noise = None
            else:
                t = torch.full((b,), self.num_timesteps - 1, device=device, dtype=torch.long)
                noise = self.diffuse_fn(coarse_mel, t) * mel_mask.unsqueeze(-1).transpose(1, -1)
            x_0_pred = self.sampling(noise=noise)[-1] * mel_mask
        else:
            mel_mask = mel_mask.unsqueeze(-1).transpose(1, -1)
            t: Tensor = torch.randint(0, self.num_timesteps, (b,), device=device).long()

            # Diffusion
            x_t = self.diffuse_fn(mel, t) * mel_mask
            x_t_prev = self.diffuse_fn(mel, t - 1) * mel_mask

            # Predict x_{start}
            x_0_pred = self.denoise_fn.forward(x_t, t, cond, spk_emb) * mel_mask
            if clip_denoised:
                x_0_pred.clamp_(-1., 1.)

            # Sample x_{t-1} using the posterior distribution
            if self.model != "shallow":
                x_start = x_0_pred
            else:
                # x_start = self.norm_spec(coarse_mel)
                x_start = coarse_mel
                # x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]
                x_start = x_start[:, None, :, :]  # [B, 1, T, M]

            x_t_prev_pred = self.q_posterior_sample(x_start=x_start, x_t=x_t, t=t) * mel_mask

            x_0_pred = x_0_pred[:, 0].transpose(1, 2)
            x_t = x_t[:, 0].transpose(1, 2)

            x_t_prev = x_t_prev[:, 0].transpose(1, 2)
            x_t_prev_pred = x_t_prev_pred[:, 0].transpose(1, 2)

        return x_0_pred, x_t, x_t_prev, x_t_prev_pred, t
