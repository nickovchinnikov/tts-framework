import torch
from torch import nn
from torch.nn import Module


class DepthWiseConv1d(Module):
    r"""Implements Depthwise 1D convolution. This module will apply a spatial convolution over inputs
    independently over each input channel in the style of depthwise convolutions.

    In a depthwise convolution, each input channel is convolved with its own set of filters, as opposed
    to standard convolutions where each input channel is convolved with all filters.
    At `groups=in_channels`, each input channel is convolved with its own set of filters.
    Filters in the
    DepthwiseConv1d are not shared among channels. This method can drastically reduce the number of
    parameters/learnable weights in the model, as each input channel gets its own filter.

    This technique is best suited to scenarios where the correlation between different channels is
    believed to be low. It is commonly employed in MobileNet models due to the reduced number of
    parameters, which is critical in mobile devices where computational resources are limited.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the convolving kernel
        padding (int): Zero-padding added to both sides of the input

    Shape:
        - Input: (N, C_in, L_in)
        - Output: (N, C_out, L_out), where

          `L_out = [L_in + 2*padding - (dilation*(kernel_size-1) + 1)]/stride + 1`

    Attributes:
        weight (Tensor): the learnable weights of shape (`out_channels`, `in_channels`/`group`, `kernel_size`)
        bias (Tensor, optional): the learnable bias of the module of shape (`out_channels`)

    Examples:
    ```python
    m = DepthWiseConv1d(16, 33, 3, padding=1)
    input = torch.randn(20, 16, 50)
    output = m(input)
    ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x: input tensor of shape (batch_size, in_channels, signal_length)

        Returns:
            output tensor of shape (batch_size, out_channels, signal_length)
        """
        return self.conv(x)


class PointwiseConv1d(Module):
    r"""Applies a 1D pointwise (aka 1x1) convolution over an input signal composed of several input
    planes, officially known as channels in this context.

    The operation implemented is also known as a "channel mixing" operation, as each output channel can be
    seen as a linear combination of input channels.

    In the simplest case, the output value of the layer with input size
    (N, C_in, L) and output (N, C_out, L_out) can be
    precisely described as:

    $$out(N_i, C_{out_j}) = bias(C_{out_j}) +
        weight(C_{out_j}, k) * input(N_i, k)$$

    where 'N' is a batch size, 'C' denotes a number of channels,
    'L' is a length of signal sequence.
    The symbol '*' in the above indicates a 1D cross-correlation operation.

    The 1D cross correlation operation "*": [Wikipedia Cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation)

    This module supports `TensorFloat32<tf32_on_ampere>`.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        stride (int): Stride of the convolution. Default: 1
        padding (int): Zero-padding added to both sides of the input. Default: 0
        bias (bool): If set to False, the layer will not learn an additive bias. Default: True
        kernel_size (int): Size of the convolving kernel. Default: 1

    Shape:
        - Input: (N, C_in, L_in)
        - Output: (N, C_out, L_out), where

          L_out = [L_in + 2*padding - (dilation*(kernel_size-1) + 1)]/stride + 1

    Attributes:
        weight (Tensor): the learnable weights of shape (out_channels, in_channels, kernel_size)
        bias (Tensor, optional): the learnable bias of the module of shape (out_channels)

    Example:
    ```python
    m = PointwiseConv1d(16, 33, 1, padding=0, bias=True)
    input = torch.randn(20, 16, 50)
    output = m(input)
    ```


    Description of parameters:
        stride (default 1): Controls the stride for the operation, which is the number of steps the convolutional
        kernel moves for each operation. A stride of 1 means that the kernel moves one step at a time and a stride
        of 2 means skipping every other step. Higher stride values can down sample the output and lead to smaller
        output shapes.

        padding (default 0): Controls the amount of padding applied to the input. By adding padding, the spatial
        size of the output can be controlled. If it is set to 0, no padding is applied. If it is set to 1, zero
        padding of one pixel width is added to the input data.

        bias (default True): Controls whether the layer uses a bias vector. By default, it is True, meaning that
        the layer has a learnable bias parameter.

        kernel_size (default 1): The size of the convolving kernel. In the case of 1D convolution, kernel_size is
        a single integer that specifies the number of elements the filter that convolves the input should have.
        In your PointwiseConv1d case, the default kernel size is 1, indicating a 1x1 convolution is applied
        which is commonly known as a pointwise convolution.

    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        kernel_size: int = 1,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Defines the computation performed at every call.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, in_channels, signal_length)

        Returns:
            output (torch.Tensor): tensor of shape (batch_size, out_channels, signal_length)
        """
        return self.conv(x)
