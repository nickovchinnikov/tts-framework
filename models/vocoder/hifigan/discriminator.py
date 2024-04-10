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

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        r"""Computes the forward pass of the discriminator.

        Args:
            x (Tensor): Input tensor of shape [B, C, T].

        Returns:
            tuple(Tensor, Tensor): Tuple containing the output tensors of the `MultiScaleDiscriminator` and `MultiPeriodDiscriminator` instances.
        """
        return self.MPD(x), self.MSD(x)
