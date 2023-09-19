import torch
from torch import nn

from model.conv_blocks import ConvTransposed
from model.constants import LEAKY_RELU_SLOPE


class VariancePredictor(nn.Module):
    r"""
    This is a Duration and Pitch predictor neural network module in PyTorch.

    It consists of multiple layers, including `ConvTransposed` layers (custom convolution transpose layers from 
    the `model.conv_blocks` module), LeakyReLU activation functions, Layer Normalization and Dropout layers.

    Constructor for `VariancePredictor` class. 

    Args:
        channels_in (int): Number of input channels.
        channels (int): Number of output channels for ConvTransposed layers and input channels for linear layer.
        channels_out (int): Number of output channels for linear layer.
        kernel_size (int): Size of the kernel for ConvTransposed layers.
        p_dropout (float): Probability of dropout.

    Returns:
        torch.Tensor: Output tensor.
    """
    def __init__(
        self,
        channels_in: int,
        channels: int,
        channels_out: int,
        kernel_size: int,
        p_dropout: float,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                # Convolution transpose layer followed by LeakyReLU, LayerNorm and Dropout
                ConvTransposed(
                    channels_in,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(leaky_relu_slope),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
                # Another "block" of ConvTransposed, LeakyReLU, LayerNorm, and Dropout
                ConvTransposed(
                    channels,
                    channels,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(leaky_relu_slope),
                nn.LayerNorm(channels),
                nn.Dropout(p_dropout),
            ]
        )

        # Output linear layer
        self.linear_layer = nn.Linear(channels, channels_out)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass for `VariancePredictor`.

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor, has the same size as x. 
            
        Returns:
            torch.Tensor: Output tensor.
        """

        # Sequentially pass the input through all defined layers 
        # (ConvTransposed -> LeakyReLU -> LayerNorm -> Dropout -> ConvTransposed -> LeakyReLU -> LayerNorm -> Dropout)
        for layer in self.layers:
            x = layer(x)
        x = self.linear_layer(x)
        x = x.squeeze(-1)
        x = x.masked_fill(mask, 0.0)
        return x
