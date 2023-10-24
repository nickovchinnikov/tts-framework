import torch
from torch.nn import Module

from model.config import VocoderModelConfig

from .multi_period_discriminator import MultiPeriodDiscriminator
from .multi_resolution_discriminator import MultiResolutionDiscriminator


class Discriminator(Module):
    r"""Discriminator for the UnuvNet vocoder.

    This class implements a discriminator that consists of a `MultiResolutionDiscriminator` and a `MultiPeriodDiscriminator`.

    Args:
        model_config (VocoderModelConfig): Model configuration object.

    Attributes:
        MRD (MultiResolutionDiscriminator): Multi-resolution discriminator instance.
        MPD (MultiPeriodDiscriminator): Multi-period discriminator instance.

    Methods:
        forward(x): Computes the forward pass of the discriminator.

    """

    def __init__(self, model_config: VocoderModelConfig):
        super().__init__()
        self.MRD = MultiResolutionDiscriminator(model_config=model_config)
        self.MPD = MultiPeriodDiscriminator(model_config=model_config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes the forward pass of the discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            tuple(torch.Tensor, torch.Tensor): Tuple containing the output tensors of the `MultiResolutionDiscriminator` and `MultiPeriodDiscriminator` instances.
        """
        return self.MRD(x), self.MPD(x)
