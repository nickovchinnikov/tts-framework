import torch
from torch import nn
from torch.nn import Module

from models.helpers import tools
from models.tts.delightful_tts.constants import LEAKY_RELU_SLOPE
from models.tts.delightful_tts.conv_blocks import (
    DepthWiseConv1d,
    GLUActivation,
    PointwiseConv1d,
)


class ConformerConvModule(Module):
    r"""Conformer Convolution Module class represents a module in the Conformer model architecture.
    The module includes a layer normalization, pointwise and depthwise convolutional layers,
    Gated Linear Units (GLU) activation, and dropout layer.

    Args:
        d_model (int): The number of expected features in the input.
        expansion_factor (int): The expansion factor for the hidden layer size in the feed-forward network, default is 2.
        kernel_size (int): The size of the convolving kernel, default is 7.
        dropout (float): The dropout probability, default is 0.1.
        leaky_relu_slope (float): Controls the angle of the negative slope of the LeakyReLU activation, default is `LEAKY_RELU_SLOPE`.
    """

    def __init__(
        self,
        d_model: int,
        expansion_factor: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()
        inner_dim = d_model * expansion_factor
        self.ln_1 = nn.LayerNorm(d_model)
        self.conv_1 = PointwiseConv1d(
            d_model,
            inner_dim * 2,
        )
        self.conv_act = GLUActivation()
        self.depthwise = DepthWiseConv1d(
            inner_dim,
            inner_dim,
            kernel_size=kernel_size,
            padding=tools.calc_same_padding(kernel_size)[0],
        )
        self.ln_2 = nn.GroupNorm(
            1,
            inner_dim,
        )
        self.activation = nn.LeakyReLU(leaky_relu_slope)
        self.conv_2 = PointwiseConv1d(
            inner_dim,
            d_model,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the Conformer conv module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, num_features).
        """
        x = self.ln_1(x)
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = self.conv_act(x)
        x = self.depthwise(x)
        x = self.ln_2(x)
        x = self.activation(x)
        x = self.conv_2(x)
        x = x.permute(0, 2, 1)
        return self.dropout(x)
