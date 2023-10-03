from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm, weight_norm

from model.basenn import BaseNNModule
from model.config import VocoderModelConfig
from model.helpers.tools import get_device


class DiscriminatorP(BaseNNModule):
    r"""
    DiscriminatorP is a class that implements a discriminator network for the UnivNet vocoder.

    Args:
        period (int): The period of the Mel spectrogram.
        model_config (VocoderModelConfig): The configuration object for the UnivNet vocoder model.
        device (torch.device, optional): The device to use for the model. Defaults to the result of `get_device()`.
    """

    def __init__(
        self,
        period: int,
        model_config: VocoderModelConfig,
        device: torch.device = get_device(),
    ):
        super().__init__(device=device)

        self.LRELU_SLOPE = model_config.mpd.lReLU_slope
        self.period = period

        kernel_size = model_config.mpd.kernel_size
        stride = model_config.mpd.stride

        norm_f: Any = (
            spectral_norm if model_config.mpd.use_spectral_norm else weight_norm
        )

        self.convs = nn.ModuleList(
            [
                norm_f(
                    nn.Conv2d(
                        1,
                        64,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        64,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        256,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        256,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(kernel_size // 2, 0),
                        device=self.device,
                    )
                ),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(
                1024,
                1,
                (3, 1),
                1,
                padding=(1, 0),
                device=self.device,
            )
        )

    def forward(self, x: torch.Tensor) -> Tuple[list, torch.Tensor]:
        r"""
        Forward pass of the discriminator network.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, channels, time_steps).

        Returns:
            Tuple[list, torch.Tensor]: A tuple containing a list of feature maps and the output tensor of shape (batch_size, period).
        """
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layers in self.convs:
            x = layers(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x
