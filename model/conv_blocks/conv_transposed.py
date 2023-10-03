import torch
import torch.nn as nn

from model.basenn import BaseNNModule
from model.helpers.tools import get_device

from .bsconv import BSConv1d


class ConvTransposed(BaseNNModule):
    r"""
    `ConvTransposed` applies a 1D convolution operation, with the main difference that it transposes the
    last two dimensions of the input tensor before and after applying the `BSConv1d` convolution operation.
    This can be useful in certain architectures where the tensor dimensions are processed in a different order.

    The `ConvTransposed` class performs a `BSConv` operation after transposing the input tensor dimensions. Specifically, it swaps the channels and width dimensions of a tensor, applies the convolution, and then swaps the dimensions back to their original order. The intuition behind swapping dimensions can depend on the specific use case in the larger architecture; typically, it's used when the operation or sequence of operations expected a different arrangement of dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the kernel used in convolution
        padding (int): Zero-padding added around the input tensor along the width direction
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`

    Attributes:
        conv (BSConv1d): `BSConv1d` module to apply convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        padding: int = 0,
        device: torch.device = get_device(),
    ):
        super().__init__(device)

        # Define BSConv1d convolutional layer
        self.conv = BSConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            device=self.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation method for the ConvTransposed layer.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            x (torch.Tensor): output tensor after application of ConvTransposed
        """

        # Transpose the last two dimensions (dimension 1 and 2 here). Now the tensor has shape (N, W, C)
        x = x.contiguous().transpose(1, 2)

        # Apply BSConv1d convolution.
        x = self.conv(x)

        # Transpose the last two dimensions back to their original order. Now the tensor has shape (N, C, W)
        x = x.contiguous().transpose(1, 2)

        # Return final output tensor
        return x
