import torch

from model.basenn import BaseNNModule
from config import VocoderModelConfig

from helpers.tools import get_device

from .multi_period_discriminator import MultiPeriodDiscriminator
from .multi_resolution_discriminator import MultiResolutionDiscriminator


class Discriminator(BaseNNModule):
    r"""
    Discriminator for the UnuvNet vocoder.

    This class implements a discriminator that consists of a `MultiResolutionDiscriminator` and a `MultiPeriodDiscriminator`.

    Args:
        model_config (VocoderModelConfig): Model configuration object.
        device (torch.device, optional): The device to use for the model. Defaults to the result of `get_device()`.

    Attributes:
        MRD (MultiResolutionDiscriminator): Multi-resolution discriminator instance.
        MPD (MultiPeriodDiscriminator): Multi-period discriminator instance.

    Methods:
        forward(x): Computes the forward pass of the discriminator.

    """

    def __init__(
        self, model_config: VocoderModelConfig, device: torch.device = get_device()
    ):
        super(Discriminator, self).__init__(device=device)
        self.MRD = MultiResolutionDiscriminator(
            model_config=model_config, device=self.device
        )
        self.MPD = MultiPeriodDiscriminator(
            model_config=model_config, device=self.device
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Computes the forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            tuple(torch.Tensor, torch.Tensor): Tuple containing the output tensors of the `MultiResolutionDiscriminator` and `MultiPeriodDiscriminator` instances.
        """
        return self.MRD(x), self.MPD(x)
