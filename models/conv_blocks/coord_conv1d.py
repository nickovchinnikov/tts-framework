import torch
from torch import nn
from torch.nn import Module
from torch.nn.modules import conv

from .add_coords import AddCoords


class CoordConv1d(conv.Conv1d, Module):
    r"""`CoordConv1d` is an extension of the standard 1D convolution layer (`conv.Conv1d`), with the addition of extra coordinate
    channels. These extra channels encode positional coordinates, and optionally, the radial distance from the origin.
    This is inspired by the paper:
    [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
    and is designed to help Convolution layers to pay attention to the absolute position of features in the input space.

    The responsibility of this class is to intercept the input tensor and append extra channels to it. These extra channels
    encode the positional coordinates (and optionally, the radial distance from the center). The enhanced tensor is then
    immediately passed through a standard Conv1D layer.

    In concrete terms, this means Convolution layer does not just process the color in an image-based task, but also 'knows'
    where in the overall image this color is located.

    In a typical Text-To-Speech (TTS) system like DelightfulTTS, the utterance is processed in a sequential manner.
    The importance of sequential data in such a use-case can benefit from `CoordConv` layer as it offers a way to draw
    more attention to the positioning of data. `CoordConv` is a drop-in replacement for standard convolution layers,
    enriches spatial representation in Convolutional Neural Networks (CNN) with additional positional information.

    Hence, the resultant Convolution does not only process the characteristics of the sound in the input speech signal,
    but also 'knows' where in the overall signal this particular sound is located, providing it with the spatial context.
    This can be particularly useful in TTS systems where the sequence of phonemes and their timing can be critical.

    Args:
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution. Default: 1.
        padding (int): Zero-padding added to both sides of the input . Default: 0.
        dilation (int): Spacing between kernel elements. Default: 1.
        groups (int): Number of blocked connections from input channels to output channels. Default: 1.
        bias (bool): If True, adds a learnable bias to the output. Default: True.
        with_r (bool): If True, adds a radial coordinate channel. Default: False.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        with_r: bool = False,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r)

        self.conv = nn.Conv1d(
            in_channels + self.rank + int(with_r),
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""The forward pass of the `CoordConv1d` module. It adds the coordinate channels to the input tensor with the `AddCoords`
        module, and then immediately passes the result through a 1D convolution.

        As a result, the subsequent Conv layers don't merely process sound characteristics of the speech signal, but are
        also aware of their relative positioning, offering a notable improvement over traditional methods, particularly for
        challenging TTS tasks where the sequence is critical.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, out_channels, length).
        """
        # Apply AddCoords layer to add coordinate channels to the input tensor
        x = self.addcoords(x)

        # Apply convolution
        return self.conv(x)
