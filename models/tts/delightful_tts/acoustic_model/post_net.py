from typing import Optional

import torch
from torch import nn


class ConvNorm(nn.Module):
    """1D Convolution

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        padding (int): Padding
        dilation (int): Dilation
        bias (bool): Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PostNet(nn.Module):
    """PostNet: Five 1-d convolution with 512 channels and kernel size 5

    Args:
        n_mel_channels (int): Number of mel channels
        postnet_embedding_dim (int): PostNet embedding dimension
        postnet_kernel_size (int): PostNet kernel size
        postnet_n_convolutions (int): Number of PostNet convolutions
        upsampling_factor (int): Upsampling factor for mel-spectrogram
        p_dropout (float): Dropout probability
    """

    def __init__(
        self,
        n_hidden: int,
        n_mel_channels: int = 100,
        postnet_embedding_dim: int = 512,
        postnet_kernel_size: int = 5,
        postnet_n_convolutions: int = 3,
        p_dropout: float = 0.1,
    ):
        super().__init__()

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(
                    n_hidden,
                    postnet_embedding_dim,
                    kernel_size=postnet_kernel_size,
                    stride=1,
                    padding=int((postnet_kernel_size - 1) / 2),
                    dilation=1,
                ),
                nn.BatchNorm1d(postnet_embedding_dim),
                nn.Dropout(p_dropout),
            ),
        )

        for _ in range(postnet_n_convolutions):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(
                        postnet_embedding_dim,
                        postnet_embedding_dim,
                        kernel_size=postnet_kernel_size,
                        stride=1,
                        padding=int((postnet_kernel_size - 1) / 2),
                        dilation=1,
                    ),
                    nn.LayerNorm(
                        postnet_embedding_dim,
                    ),
                    nn.Dropout(p_dropout),
                ),
            )

        self.to_mel = nn.Linear(
            postnet_embedding_dim,
            n_mel_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convolutions:
            x = conv(x)

        return self.to_mel(x)
