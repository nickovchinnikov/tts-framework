from typing import Any, Tuple

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from models.config import VocoderModelConfig


class DiscriminatorP(Module):
    r"""DiscriminatorP is a class that implements a discriminator network for the UnivNet vocoder.

    Args:
        period (int): The period of the Mel spectrogram.
        model_config (VocoderModelConfig): The configuration object for the UnivNet vocoder model.
    """

    def __init__(
        self,
        period: int,
        model_config: VocoderModelConfig,
    ):
        super().__init__()

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
                    ),
                ),
                norm_f(
                    nn.Conv2d(
                        64,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    ),
                ),
                norm_f(
                    nn.Conv2d(
                        128,
                        256,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    ),
                ),
                norm_f(
                    nn.Conv2d(
                        256,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(kernel_size // 2, 0),
                    ),
                ),
                norm_f(
                    nn.Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(kernel_size // 2, 0),
                    ),
                ),
            ],
        )
        self.conv_post = norm_f(
            nn.Conv2d(
                1024,
                1,
                (3, 1),
                1,
                padding=(1, 0),
            ),
        )

    def forward(self, x: torch.Tensor) -> Tuple[list, torch.Tensor]:
        r"""Forward pass of the discriminator network.

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
            x = layers(x.to(dtype=self.conv_post.weight.dtype))
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return fmap, x
