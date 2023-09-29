import torch
import torch.nn as nn

from model.helpers.tools import get_device
from model.config import VocoderModelConfig

from model.basenn import BaseNNModule

from .discriminator_r import DiscriminatorR


class MultiResolutionDiscriminator(BaseNNModule):
    r"""
    Multi-resolution discriminator for the UnivNet vocoder.

    This class implements a multi-resolution discriminator that consists of multiple DiscriminatorR instances, each operating at a different resolution.

    Args:
        model_config (VocoderModelConfig): Model configuration object.
        device (torch.device, optional): The device to use for the model. Defaults to the result of `get_device()`.

    Attributes:
        resolutions (list): List of resolutions for each DiscriminatorR instance.
        discriminators (nn.ModuleList): List of DiscriminatorR instances.

    Methods:
        forward(x): Computes the forward pass of the multi-resolution discriminator.

    """

    def __init__(
        self, model_config: VocoderModelConfig, device: torch.device = get_device()
    ):
        super(MultiResolutionDiscriminator, self).__init__(device=device)

        self.resolutions = model_config.mrd.resolutions
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorR(
                    resolution, model_config=model_config, device=self.device
                )
                for resolution in self.resolutions
            ]
        )

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        r"""
        Computes the forward pass of the multi-resolution discriminator.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, T].

        Returns:
            list: List of tuples containing the intermediate feature maps and the output scores for each `DiscriminatorR` instance.
        """
        ret = list()
        for disc in self.discriminators:
            ret.append(disc(x))

        return ret  # [(feat, score), (feat, score), (feat, score)]
