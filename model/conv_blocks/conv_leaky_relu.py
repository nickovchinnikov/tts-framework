from lightning.pytorch import LightningModule
import torch.nn as nn

from model.constants import LEAKY_RELU_SLOPE


# TODO: prepared for the refactoring of Aligner
class ConvLeakyReLU(LightningModule):
    r"""
    This class implements a Convolution followed by a Leaky ReLU activation layer.

    Attributes
    ----------
    layers : nn.Sequential
        Sequential container that holds the Convolution and LeakyReLU layers.

    Methods
    -------
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
        r"""
        Parameters
        ----------
        in_channels : int
            Number of channels in the input.
        out_channels : int
            Number of channels in the output.
        kernel_size : int
            Size of the convolving kernel.
        padding : int
            Zero-padding added to both sides of the input.
        leaky_relu_slope : float
            Controls the angle of the negative slope, default=LEAKY_RELU_SLOPE.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size, padding=padding
            ),
            nn.LeakyReLU(leaky_relu_slope),
        )

    def forward(self, x):
        r"""
        Defines the forward pass of the ConvLeakyReLU.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor after being passed through the Conv1d and LeakyReLU layers.
        """
        return self.layers(x)
