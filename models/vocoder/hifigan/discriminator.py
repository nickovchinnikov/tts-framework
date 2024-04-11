from typing import List, Tuple

from torch import Tensor
from torch.nn import Module

from .mp_discriminator import MultiPeriodDiscriminator
from .ms_discriminator import MultiScaleDiscriminator


class Discriminator(Module):
    r"""Discriminator for the HifiGan vocoder.

    This class implements a discriminator that consists of a `MultiPeriodDiscriminator` and a `MultiScaleDiscriminator`.

    Args:
        model_config (VocoderModelConfig): Model configuration object.

    Attributes:
        MPD (MultiPeriodDiscriminator): Multi-resolution discriminator instance.
        MSD (MultiScaleDiscriminator): Multi-scale discriminator instance.

    Methods:
        forward(x): Computes the forward pass of the discriminator.

    """

    def __init__(self):
        super().__init__()
        self.MPD = MultiScaleDiscriminator()
        self.MSD = MultiPeriodDiscriminator()

    def forward(
        self,
        x: Tensor,
        x_hat: Tensor,
    ) -> Tuple[
        Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
        Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
    ]:
        r"""Computes the forward pass of the discriminator.

        Args:
            x (Tensor): Input tensor of shape [B, C, T].
            x_hat (Tensor): Input tensor of shape [B, C, T].

        Returns:
            Tuple[
                Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
                Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
            ]:
            Tuple containing the output tensors of the `MultiResolutionDiscriminator` and `MultiPeriodDiscriminator` instances.
        """
        return self.MPD.forward(x, x_hat), self.MSD.forward(x, x_hat)
