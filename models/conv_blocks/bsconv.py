import torch
from torch.nn import Module

from .conv1d import DepthWiseConv1d, PointwiseConv1d


class BSConv1d(Module):
    r"""`BSConv1d` implements the `BSConv` concept which is based on the paper [BSConv:
    Binarized Separated Convolutional Neural Networks](https://arxiv.org/pdf/2003.13549.pdf).

    `BSConv` is an amalgamation of depthwise separable convolution and pointwise convolution.
    Depthwise separable convolution utilizes far fewer parameters by separating the spatial
    (depthwise) and channel-wise (pointwise) operations. Meanwhile, pointwise convolution
    helps in transforming the channel characteristics without considering the channel's context.

    Args:
        channels_in (int): Number of input channels
        channels_out (int): Number of output channels produced by the convolution
        kernel_size (int): Size of the kernel used in depthwise convolution
        padding (int): Zeropadding added around the input tensor along the height and width directions

    Attributes:
        pointwise (PointwiseConv1d): Pointwise convolution module
        depthwise (DepthWiseConv1d): Depthwise separable convolution module
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()

        # Instantiate Pointwise Convolution Module:
        # First operation in BSConv: the number of input channels is transformed to the number
        # of output channels without taking into account the channel context.
        self.pointwise = PointwiseConv1d(channels_in, channels_out)

        # Instantiate Depthwise Convolution Module:
        # Second operation in BSConv: A spatial convolution is performed independently over each output
        # channel from the pointwise convolution.
        self.depthwise = DepthWiseConv1d(
            channels_out,
            channels_out,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Propagate input tensor through pointwise convolution.
        x1 = self.pointwise(x)

        # Propagate the result of the previous pointwise convolution through the depthwise convolution.
        # Return final output of the sequence of pointwise and depthwise convolutions
        return self.depthwise(x1)
