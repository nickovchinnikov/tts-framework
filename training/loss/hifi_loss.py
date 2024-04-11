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
    loss = torch.tensor(0)

    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


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
    total_loss = torch.tensor(0)

    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        total_loss += r_loss + g_loss

    return total_loss


def generator_loss(disc_outputs: List[Tensor]):
    r"""Calculates the generator loss.

    Args:
        disc_outputs (List[Tensor]): List of discriminator's outputs for generated inputs.

    Returns:
        Tensor: The total loss and list of individual losses.
    """
    total_loss = torch.tensor(0)
    for dg in disc_outputs:
        total_loss += torch.mean((1 - dg) ** 2)

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
        res_fake: List[Tuple[Tensor, Tensor]],
        period_fake: List[Tuple[Tensor, Tensor]],
        res_real: List[Tuple[Tensor, Tensor]],
        period_real: List[Tuple[Tensor, Tensor]],
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        r"""Calculate the losses for the generator and discriminator.

        Args:
            audio (torch.Tensor): The real audio samples.
            fake_audio (torch.Tensor): The generated audio samples.
            res_fake (List[Tuple[Tensor, Tensor]]): The discriminator's output for the fake audio.
            period_fake (List[Tuple[Tensor, Tensor]]): The discriminator's output for the fake audio in the period.
            res_real (List[Tuple[Tensor, Tensor]]): The discriminator's output for the real audio.
            period_real (List[Tuple[Tensor, Tensor]]): The discriminator's output for the real audio in the period.

        Returns:
            tuple: A tuple containing the univnet loss, discriminator loss, STFT loss and score loss.
        """
        # Calculate the STFT loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * self.stft_lamb

        # Calculate the score loss
        score_loss = torch.tensor(0.0, device=audio.device)
        for _, score_fake in res_fake + period_fake:
            score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

        score_loss = score_loss / len(res_fake + period_fake)

        # Calculate the total generator loss
        total_loss_gen = score_loss + stft_loss

        # Calculate the discriminator loss
        total_loss_disc = torch.tensor(0.0, device=audio.device)
        for (_, score_fake), (_, score_real) in zip(
            res_fake + period_fake,
            res_real + period_real,
        ):
            total_loss_disc += torch.mean(torch.pow(score_real - 1.0, 2)) + torch.mean(
                torch.pow(score_fake, 2),
            )

        total_loss_disc = total_loss_disc / len(res_fake + period_fake)

        return (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
        )
