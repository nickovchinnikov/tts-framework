import torch
from torch import Tensor, nn
from torch.nn import functional as F


class AdaLayerNorm(nn.Module):
    r"""A class used to represent an adaptive layer normalization module.

    Attributes:
        channels (int): The number of channels in the input data.
        eps (float): A small value added to the denominator for numerical stability.
        fc (nn.Linear): A fully connected layer used to compute the scale and shift parameters.

    Args:
        style_dim (int): The dimension of the style vector.
        channels (int): The number of channels in the input data.
        eps (float, optional): A small value added to the denominator for numerical stability. Defaults to 1e-5.
    """

    def __init__(self, style_dim: int, channels: int, eps: float=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.fc = nn.Linear(style_dim, channels*2)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        r"""Applies adaptive layer normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, num_samples, num_channels).
            s (torch.Tensor): The style tensor of shape (batch_size, style_dim).

        Returns:
            torch.Tensor: The normalized tensor of the same shape as the input tensor.
        """
        x = x.transpose(-1, -2)
        x = x.transpose(1, -1)

        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1)

        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta
        return x.transpose(1, -1).transpose(-1, -2)
