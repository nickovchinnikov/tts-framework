import torch
from torch import nn
from torch.nn import Module

from models.config import VocoderModelConfig

from .discriminator_r import DiscriminatorR


class MultiResolutionDiscriminator(Module):
    r"""Multi-resolution discriminator for the UnivNet vocoder.

    This class implements a multi-resolution discriminator that consists of multiple DiscriminatorR instances, each operating at a different resolution.

    Args:
        model_config (VocoderModelConfig): Model configuration object.

    Attributes:
        resolutions (list): List of resolutions for each DiscriminatorR instance.
        discriminators (nn.ModuleList): List of DiscriminatorR instances.

    Methods:
        forward(x): Computes the forward pass of the multi-resolution discriminator.

    """

    def __init__(
        self,
        model_config: VocoderModelConfig,
    ):
        super().__init__()

        self.resolutions = model_config.mrd.resolutions
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(resolution, model_config=model_config)
                for resolution in self.resolutions
            ],
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        r"""Computes the forward pass of the multi-resolution discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            list: List of tuples containing the intermediate feature maps and the output scores for each `DiscriminatorR` instance.
        """
        return [disc(x) for disc in self.discriminators] # [(feat, score), (feat, score), (feat, score)]
