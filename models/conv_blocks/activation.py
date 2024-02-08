import torch
from torch import nn
from torch.nn import Module

from models.constants import LEAKY_RELU_SLOPE


class GLUActivation(Module):
    r"""Implements the Gated Linear Unit (GLU) activation function.

    The GLU activation splits the input in half across the channel dimension.
    One half is passed through a nonlinear activation function (like sigmoid or leaky ReLU),
    and the output from this activation function is used as a gate to control the
    amplitude of the other half of the input. An element-wise multiplication is then performed
    between the gating signal and the other half of the input.

    The GLU activation allows the model to dynamically choose which inputs to pass through and
    what information to suppress, which can help improving the model performance on certain tasks.

    Args:
        slope: Controls the slope for the leaky ReLU activation function. Default: 0.3 or see the const `LEAKY_RELU_SLOPE`

    Shape:
        - Input: (N, 2*C, L) where C is the number of input channels.
        - Output: (N, C, L)

    Examples:
    ```python
    m = GLUActivation(0.3)
    input = torch.randn(16, 2*20, 44)
    output = m(input)
    ```

    """

    def __init__(self, slope: float = LEAKY_RELU_SLOPE):
        super().__init__()
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call.

        Args:
            x: The input tensor of shape (batch_size, 2*channels, signal_length)

        Returns:
            x: The output tensor of shape (batch_size, channels, signal_length)
        """
        # Split the input into two equal parts (chunks) along dimension 1
        out, gate = x.chunk(2, dim=1)

        # Perform element-wise multiplication of the first half (out)
        # with the result of applying LeakyReLU on the second half (gate)
        return out * self.lrelu(gate)
