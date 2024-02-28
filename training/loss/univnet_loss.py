import torch
from torch.nn import Module

from models.config import VocoderBasicConfig, VocoderModelConfig

from .multi_resolution_stft_loss import MultiResolutionSTFTLoss


class UnivnetLoss(Module):
    r"""UnivnetLoss is a PyTorch Module that calculates the generator and discriminator losses for Univnet."""

    def __init__(self):
        r"""Initializes the UnivnetLoss module."""
        super().__init__()

        train_config = VocoderBasicConfig()

        self.stft_lamb = train_config.stft_lamb
        self.model_config = VocoderModelConfig()

        self.stft_criterion = MultiResolutionSTFTLoss(self.model_config.mrd.resolutions)

    def forward(
        self,
        audio: torch.Tensor,
        fake_audio: torch.Tensor,
        res_fake: torch.Tensor,
        period_fake: torch.Tensor,
        res_real: torch.Tensor,
        period_real: torch.Tensor,
    ):
        r"""Calculate the losses for the generator and discriminator.

        Args:
            audio (torch.Tensor): The real audio samples.
            fake_audio (torch.Tensor): The generated audio samples.
            res_fake (torch.Tensor): The discriminator's output for the fake audio.
            period_fake (torch.Tensor): The discriminator's output for the fake audio in the period.
            res_real (torch.Tensor): The discriminator's output for the real audio.
            period_real (torch.Tensor): The discriminator's output for the real audio in the period.

        Returns:
            tuple: A tuple containing the univnet loss, discriminator loss, STFT loss, and score loss.
        """
        # Calculate the STFT loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * self.stft_lamb

        # Calculate the score loss
        score_loss = torch.tensor(0.0, device=audio.device)
        for (_, score_fake) in res_fake + period_fake:
            score_loss += torch.mean(torch.pow(score_fake - 1.0, 2))

        score_loss = score_loss / len(res_fake + period_fake)

        # Calculate the total generator loss
        total_loss_gen = score_loss + stft_loss

        # Calculate the discriminator loss
        total_loss_disc = torch.tensor(0.0, device=audio.device)
        for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real):
            total_loss_disc += torch.mean(torch.pow(score_real - 1.0, 2)) + \
                torch.mean(torch.pow(score_fake, 2))

        total_loss_disc = total_loss_disc / len(res_fake + period_fake)

        return (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
        )
