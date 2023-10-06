from lightning.pytorch import LightningModule
import torch
import torch.nn as nn

from model.config import VocoderModelConfig

from .discriminator_p import DiscriminatorP


class MultiPeriodDiscriminator(LightningModule):
    r"""
    MultiPeriodDiscriminator is a class that implements a multi-period discriminator network for the UnivNet vocoder.

    Args:
        model_config (VocoderModelConfig): The configuration object for the UnivNet vocoder model.
    """

    def __init__(
        self,
        model_config: VocoderModelConfig,
    ):
        super().__init__()

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(period, model_config=model_config)
                for period in model_config.mpd.periods
            ]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        r"""
        Forward pass of the multi-period discriminator network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, time_steps).

        Returns:
            list: A list of output tensors from each discriminator network.
        """
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret
