import torch
import torch.nn as nn

from model.constants import LEAKY_RELU_SLOPE


class FeedForward(nn.Module):
    r"""
    Creates a feed-forward neural network.
    The network includes a layer normalization, an activation function (LeakyReLU), and dropout layers.

    Args:
        d_model (int): The number of expected features in the input.
        kernel_size (int): The size of the convolving kernel for the first conv1d layer.
        dropout (float): The dropout probability.
        expansion_factor (int, optional): The expansion factor for the hidden layer size in the feed-forward network, default is 4.
        leaky_relu_slope (float, optional): Controls the angle of the negative slope of LeakyReLU activation, default is `LEAKY_RELU_SLOPE`.
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        dropout: float,
        expansion_factor: int = 4,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)
        self.conv_1 = nn.Conv1d(
            d_model,
            d_model * expansion_factor,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.act = nn.LeakyReLU(leaky_relu_slope)
        self.conv_2 = nn.Conv1d(d_model * expansion_factor, d_model, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the feed-forward neural network.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, num_features).
        """
        # Apply layer normalization
        x = self.ln(x)

        # Forward pass through the first convolution layer, activation layer and dropout layer
        x = x.permute((0, 2, 1))
        x = self.conv_1(x)
        x = x.permute((0, 2, 1))
        x = self.act(x)
        x = self.dropout(x)

        # Forward pass through the second convolution layer and dropout layer
        x = x.permute((0, 2, 1))
        x = self.conv_2(x)
        x = x.permute((0, 2, 1))
        x = self.dropout(x)

        # Scale the output by 0.5 (this helps with training stability)
        x = 0.5 * x
        return x
