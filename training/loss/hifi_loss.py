from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from models.config import VocoderBasicConfig, VocoderModelConfig

from .multi_resolution_stft_loss import MultiResolutionSTFTLoss


def feature_loss(fmap_r: List[Tensor], fmap_g: List[Tensor]) -> Tensor:
    r"""Calculates the feature loss between real and generated feature maps.

    Args:
        fmap_r (List[Tensor]): List of real feature maps.
        fmap_g (List[Tensor]): List of generated feature maps.

    Returns:
        Tensor: The calculated feature loss.
    """
    total_loss = torch.tensor(0.0)

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            total_loss += torch.mean(torch.abs(rl - gl)).to(total_loss.device)

    return total_loss * 2


def discriminator_loss(
    disc_real_outputs: List[Tensor],
    disc_generated_outputs: List[Tensor],
) -> Tensor:
    r"""Calculates the discriminator loss for real and generated outputs.

    Args:
        disc_real_outputs (List[Tensor]): List of discriminator's outputs for real inputs.
        disc_generated_outputs (List[Tensor]): List of discriminator's outputs for generated inputs.

    Returns:
        Tensor: The discriminator loss.
    """
    total_loss = torch.tensor(0.0)

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2).to(total_loss.device)
        g_loss = torch.mean(dg**2).to(total_loss.device)
        total_loss += r_loss + g_loss

    return total_loss


def generator_loss(disc_outputs: List[Tensor]):
    r"""Calculates the generator loss.

    Args:
        disc_outputs (List[Tensor]): List of discriminator's outputs for generated inputs.

    Returns:
        Tensor: The total loss and list of individual losses.
    """
    total_loss = torch.tensor(0.0)
    for dg in disc_outputs:
        total_loss += torch.mean((1 - dg) ** 2).to(total_loss.device)

    return total_loss


class HifiLoss(Module):
    r"""HifiLoss is a PyTorch Module that calculates the generator and discriminator losses for Hifi vocoder."""

    def __init__(self):
        r"""Initializes the HifiLoss module."""
        super().__init__()

        train_config = VocoderBasicConfig()

        self.stft_lamb = train_config.stft_lamb
        self.model_config = VocoderModelConfig()

        self.stft_criterion = MultiResolutionSTFTLoss(self.model_config.mrd.resolutions)

    def forward(
        self,
        audio: Tensor,
        fake_audio: Tensor,
        mpd_res: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
        msd_res: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""Calculate the losses for the generator and discriminator.

        Args:
            audio (Tensor): The real audio samples.
            fake_audio (Tensor): The generated audio samples.
            mpd_res (Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]): The multi-resolution discriminator results for the real and generated audio.
            msd_res (Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]): The multi-scale discriminator results for the real and generated audio.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: The total discriminator loss, discriminator loss for the real and generated audio, total generator loss, generator loss for the real and generated audio, feature loss for the real and generated audio, and the STFT loss.
        """
        # Calculate the STFT loss
        sc_loss, mag_loss = self.stft_criterion.forward(
            fake_audio.squeeze(1),
            audio.squeeze(1),
        )
        stft_loss = (sc_loss + mag_loss) * self.stft_lamb

        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd_res
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd_res

        loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g).to(audio.device)
        loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g).to(audio.device)

        # Calculate the total discriminator loss
        total_loss_disc = loss_disc_f + loss_disc_s

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g).to(audio.device)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g).to(audio.device)

        loss_gen_f = generator_loss(y_df_hat_g).to(audio.device)
        loss_gen_s = generator_loss(y_ds_hat_g).to(audio.device)

        # Calculate the total generator loss
        total_loss_gen = loss_gen_f + loss_gen_s + loss_fm_s + loss_fm_f + stft_loss

        return (
            total_loss_disc,
            loss_disc_s,
            loss_disc_f,
            total_loss_gen,
            loss_gen_f,
            loss_gen_s,
            loss_fm_s,
            loss_fm_f,
            stft_loss,
        )
