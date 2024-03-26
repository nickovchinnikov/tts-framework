from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Conv2d, Module
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from models.config import HifiGanPretrainingConfig

from .utils import get_padding

# Leaky ReLU slope
LRELU_SLOPE = HifiGanPretrainingConfig.lReLU_slope


class DiscriminatorP(Module):
    def __init__(
        self,
        period: int,
        kernel_size: int = 5,
        stride: int = 3,
        use_spectral_norm: bool = False,
    ):
        r"""Initialize the DiscriminatorP module.

        Args:
            period (int): The period for the discriminator.
            kernel_size (int, optional): The kernel size for the convolutional layers. Defaults to 5.
            stride (int, optional): The stride for the convolutional layers. Defaults to 3.
            use_spectral_norm (bool, optional): Whether to use spectral normalization. Defaults to False.
        """
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    ),
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    ),
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    ),
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(get_padding(5, 1), 0),
                    ),
                ),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ],
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        r"""Forward pass of the DiscriminatorP module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tuple[Tensor, List[Tensor]]: The output tensor and a list of feature maps.
        """
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        r"""Initialize the MultiPeriodDiscriminator module."""
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ],
        )

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor,
    ) -> Tuple[
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
        List[torch.Tensor],
    ]:
        r"""Forward pass of the MultiPeriodDiscriminator module.

        Args:
            y (torch.Tensor): The real audio tensor.
            y_hat (torch.Tensor): The generated audio tensor.

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            A tuple containing lists of discriminator outputs and feature maps for real and generated audio.
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for _, discriminator in enumerate(self.discriminators):
            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
