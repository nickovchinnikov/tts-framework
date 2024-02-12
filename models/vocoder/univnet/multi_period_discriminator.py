import torch
from torch import nn
from torch.nn import Module

from models.config import VocoderModelConfig

from .discriminator_p import DiscriminatorP


class MultiPeriodDiscriminator(Module):
    r"""MultiPeriodDiscriminator is a class that implements a multi-period discriminator network for the UnivNet vocoder.

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
            ],
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        r"""Forward pass of the multi-period discriminator network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, time_steps).

        Returns:
            list: A list of output tensors from each discriminator network.
        """
        return [disc(x) for disc in self.discriminators]
