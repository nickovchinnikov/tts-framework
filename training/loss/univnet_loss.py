import torch
from torch.nn import Module

from model.config import VocoderModelConfig, VoicoderTrainingConfig

from .multi_resolution_stft_loss import MultiResolutionSTFTLoss


class UnivnetLoss(Module):
    r"""UnivnetLoss is a PyTorch Module that calculates the generator and discriminator losses for Univnet."""

    def __init__(self, train_config: VoicoderTrainingConfig):
        r"""Initializes the UnivnetLoss module.

        Args:
            train_config (VoicoderTrainingConfig): The training configuration containing the stft_lamb parameter.
        """
        super().__init__()

        self.stft_lamb = train_config.stft_lamb
        self.model_config = VocoderModelConfig()

        self.stft_criterion = MultiResolutionSTFTLoss(self.model_config.mrd.resolutions)

    def forward(
        self,
        audio: torch.Tensor,fake_audio: torch.Tensor,
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
            dict: A dictionary containing the total generator loss, total discriminator loss, mel loss, and score loss.
        """
        # Calculate the STFT loss
        sc_loss, mag_loss = self.stft_criterion(fake_audio.squeeze(1), audio.squeeze(1))
        stft_loss = (sc_loss + mag_loss) * self.stft_lamb

        # Calculate the score loss
        score_loss = sum([
            torch.mean(torch.pow(score_fake - 1.0, 2)).item()
            for (_, score_fake) in res_fake + period_fake
        ])
        score_loss = score_loss / len(res_fake + period_fake)

        # Calculate the total generator loss
        total_loss_gen = score_loss + stft_loss

        # Calculate the discriminator loss
        total_loss_disc: float = sum([
            torch.mean(torch.pow(score_real - 1.0, 2)).item() +
            torch.mean(torch.pow(score_fake, 2)).item()
            for (_, score_fake), (_, score_real) in zip(res_fake + period_fake, res_real + period_real)
        ])

        total_loss_disc = total_loss_disc / len(res_fake + period_fake)

        return {
            "total_loss_gen": total_loss_gen.item(),
            "total_loss_disc": total_loss_disc,
            "mel_loss": stft_loss.item(),
            "score_loss": score_loss,
        }
