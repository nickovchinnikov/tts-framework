import torch
import torch.nn as nn

from config import VocoderModelConfig

from model.basenn import BaseNNModule
from model.univnet import DiscriminatorP

from helpers.tools import get_device


class MultiPeriodDiscriminator(BaseNNModule):
    r"""
    MultiPeriodDiscriminator is a class that implements a multi-period discriminator network for the UnivNet vocoder.

    Args:
        model_config (VocoderModelConfig): The configuration object for the UnivNet vocoder model.
        device (torch.device, optional): The device to use for the model. Defaults to the result of `get_device()`.
    """

    def __init__(
        self, model_config: VocoderModelConfig, device: torch.device = get_device()
    ):
        super(MultiPeriodDiscriminator, self).__init__(device=device)

        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(period, model_config=model_config, device=self.device)
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