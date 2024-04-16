from typing import List, Tuple

from piq import SSIMLoss
import torch
from torch import Tensor, nn
from torch.nn import Module

from models.config import VocoderModelConfig

from .multi_resolution_stft_loss import MultiResolutionSTFTLoss
from .utils import sample_wise_min_max


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
            total_loss += torch.mean(
                torch.abs(rl - gl),
            ).to(total_loss.device)

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

        self.model_config = VocoderModelConfig()
        self.stft_criterion = MultiResolutionSTFTLoss(self.model_config.mrd.resolutions)
        self.ssim_loss = SSIMLoss()
        self.mae_loss = nn.L1Loss()

    def forward(
        self,
        audio: Tensor,
        fake_audio: Tensor,
        mel: Tensor,
        fake_mel: Tensor,
        mpd_res: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
        msd_res: Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        r"""Calculate the losses for the generator and discriminator.

        Args:
            audio (Tensor): The real audio samples.
            fake_audio (Tensor): The generated audio samples.
            mpd_res (Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]): The multi-resolution discriminator results for the real and generated audio.
            msd_res (Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]): The multi-scale discriminator results for the real and generated audio.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]: The total discriminator loss, discriminator loss for the real and generated audio, total generator loss, generator loss for the real and generated audio, feature loss for the real and generated audio, and the STFT loss.
        """
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd_res
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd_res

        # Calculate the STFT loss
        sc_loss, mag_loss = self.stft_criterion.forward(
            fake_audio.squeeze(1),
            audio.squeeze(1),
        )
        stft_loss = sc_loss + mag_loss

        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g).to(audio.device)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g).to(audio.device)

        loss_gen_f = generator_loss(y_df_hat_g).to(audio.device)
        loss_gen_s = generator_loss(y_ds_hat_g).to(audio.device)

        # SSIM loss
        mel_predictions_normalized = (
            sample_wise_min_max(fake_mel).float().to(fake_mel.device)
        )
        mel_targets_normalized = sample_wise_min_max(mel).float().to(mel.device)

        ssim_loss: torch.Tensor = self.ssim_loss(
            mel_predictions_normalized.unsqueeze(1),
            mel_targets_normalized.unsqueeze(1),
        )

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            ssim_loss = torch.tensor([1.0], device=fake_mel.device)

        loss_mel = self.mae_loss(mel, fake_mel)

        # Calculate the total generator loss
        total_loss_gen = (
            loss_gen_f
            + loss_gen_s
            + loss_fm_s
            + loss_fm_f
            + stft_loss
            + ssim_loss
            + loss_mel
        )

        loss_disc_f = discriminator_loss(y_df_hat_r, y_df_hat_g).to(audio.device)
        loss_disc_s = discriminator_loss(y_ds_hat_r, y_ds_hat_g).to(audio.device)

        # Calculate the total discriminator loss
        total_loss_disc = loss_disc_f + loss_disc_s

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
            ssim_loss,
            loss_mel,
        )
