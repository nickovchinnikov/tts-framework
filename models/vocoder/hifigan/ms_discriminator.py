from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import AvgPool1d, Conv1d, Module
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from models.config import HifiGanPretrainingConfig

# Leaky ReLU slope
LRELU_SLOPE = HifiGanPretrainingConfig.lReLU_slope


class DiscriminatorS(Module):
    def __init__(self, use_spectral_norm: bool = False):
        r"""Initialize the DiscriminatorS module.

        Args:
            use_spectral_norm (bool, optional): Whether to use spectral normalization. Defaults to False.
        """
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ],
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        r"""Forward pass of the DiscriminatorS module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tuple[Tensor, List[Tensor]]: The output tensor and a list of feature maps.
        """
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(Module):
    def __init__(self):
        r"""Initialize the MultiScaleDiscriminator module."""
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ],
        )
        self.meanpools = nn.ModuleList(
            [
                AvgPool1d(4, 2, padding=2),
                AvgPool1d(4, 2, padding=2),
            ],
        )

    def forward(
        self,
        y: Tensor,
        y_hat: Tensor,
    ) -> Tuple[
        List[Tensor],
        List[Tensor],
        List[Tensor],
        List[Tensor],
    ]:
        r"""Forward pass of the MultiScaleDiscriminator module.

        Args:
            y (Tensor): The real audio tensor.
            y_hat (Tensor): The generated audio tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
            A tuple containing lists of discriminator outputs and feature maps for real and generated audio.
        """
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for i, discriminator in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)

            y_d_r, fmap_r = discriminator(y)
            y_d_g, fmap_g = discriminator(y_hat)

            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
