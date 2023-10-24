import torch
from torch import nn
from torch.nn import Module

from model.constants import LEAKY_RELU_SLOPE


# TODO: prepared for the refactoring of Aligner
class ConvLeakyReLU(Module):
    r"""Class implements a Convolution followed by a Leaky ReLU activation layer.

    Attributes
        layers (nn.Sequential): Sequential container that holds the Convolution and LeakyReLU layers.

    Methods
    forward(x: torch.Tensor) -> torch.Tensor
        Passes the input through the Conv1d and LeakyReLU layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        r"""Args:
        in_channels (int): The number of channels in the input data. This could refer to different color channels (like RGB in an image) or different input features in a dataset.

        out_channels (int): The number of channels in the output data. This typically corresponds to the number of filters applied on the input.

        kernel_size (int): The size of the convolving kernel used in the convolution operation. This is usually an odd integer.

        padding (int): The number of zero-padding pixels added on each side of the input data. This is used to control the spatial dimensions of the output data.

        leaky_relu_slope (float, default=LEAKY_RELU_SLOPE): The slope of the function for negative values in a Leaky ReLU activation function. This controls the amount of "leakiness" or the degree to which the function allows negative values to pass through.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding,
            ),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Defines the forward pass of the ConvLeakyReLU.

        Args:
        x (torch.Tensor): The input tensor.

        Returns:
        torch.Tensor: The output tensor after being passed through the Conv1d and LeakyReLU layers.
        """
        return self.layers(x)
