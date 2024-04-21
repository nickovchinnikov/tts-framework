from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Conv1d, ConvTranspose1d, Module
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

from models.config import HifiGanConfig, HifiGanPretrainingConfig, PreprocessingConfig

from .utils import get_padding, init_weights

# Leaky ReLU slope
LRELU_SLOPE = HifiGanPretrainingConfig.lReLU_slope


class ResBlock1(Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: List[int] = [1, 3, 5],
    ):
        r"""Initialize the ResBlock1 module.

        Args:
            channels (int): The number of channels for the ResBlock.
            kernel_size (int, optional): The kernel size for the convolutional layers. Defaults to 3.
            dilation (Tuple[int, int, int], optional): The dilation for the convolutional layers. Defaults to (1, 3, 5).
        """
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    ),
                ),
            ],
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    ),
                ),
            ],
        )
        self.convs2.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the ResBlock1 module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        r"""Remove the weight normalization from the convolutional layers."""
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class ResBlock2(Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: List[int] = [1, 3],
    ):
        r"""Initialize the ResBlock2 module.

        Args:
            channels (int): The number of channels for the ResBlock.
            kernel_size (int, optional): The kernel size for the convolutional layers. Defaults to 3.
            dilation (Tuple[int, int], optional): The dilation for the convolutional layers. Defaults to (1, 3).
        """
        super().__init__()
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    ),
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    ),
                ),
            ],
        )
        self.convs.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the ResBlock2 module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        for layer in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = layer(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        r"""Remove the weight normalization from the convolutional layers."""
        for layer in self.convs:
            remove_weight_norm(layer)


class Generator(Module):
    def __init__(self, h: HifiGanConfig, p: PreprocessingConfig):
        r"""Initialize the Generator module.

        Args:
            h (HifiGanConfig): The configuration for the Generator.
            p (PreprocessingConfig): The configuration for the preprocessing.
        """
        super().__init__()
        self.h = h
        self.p = p
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(
            Conv1d(
                self.p.stft.n_mel_channels,
                h.upsample_initial_channel,
                7,
                1,
                padding=3,
            ),
        )
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    ),
                ),
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            resblock_list = nn.ModuleList()
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes),
            ):
                resblock_list.append(resblock(ch, k, d))
            self.resblocks.append(resblock_list)

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the Generator module.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = torch.zeros_like(x)

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x
