import torch
from torch import nn
from torch.nn import Module

from .bsconv import BSConv1d


class Conv1dGLU(Module):
    r"""`Conv1dGLU` implements a variant of Convolutional Layer with a Gated Linear Unit (GLU).
    It's based on the Deep Voice 3 project.

    Args:
        d_model (int): model dimension parameter.
        kernel_size (int): kernel size for the convolution layer.
        padding (int): padding size for the convolution layer.
        embedding_dim (int): dimension of the embedding.

    Attributes:
         bsconv1d (BSConv1d) : an instance of the Binarized Separated Convolution (1d)
         embedding_proj (torch.nn.Modules.Linear): linear transformation for embeddings.
         sqrt (torch.Tensor): buffer that stores the square root of 0.5
         softsign (torch.nn.SoftSign): SoftSign Activation function
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int,
        padding: int,
        embedding_dim: int,
    ):
        super().__init__()

        self.bsconv1d = BSConv1d(
            d_model,
            2 * d_model,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.embedding_proj = nn.Linear(
            embedding_dim,
            d_model,
        )

        self.register_buffer("sqrt", torch.sqrt(torch.tensor([0.5])).squeeze(0))

        self.softsign = torch.nn.Softsign()

    def forward(self, x: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward propagation method for the Conv1dGLU layer.

        Args:
            x (torch.Tensor): input tensor
            embeddings (torch.Tensor): input embeddings

        Returns:
            x (torch.Tensor): output tensor after application of Conv1dGLU
        """
        x = x.permute((0, 2, 1))
        residual = x
        x = self.bsconv1d(x)
        splitdim = 1
        a, b = x.split(x.size(splitdim) // 2, dim=splitdim)
        embeddings = self.embedding_proj(embeddings)
        softsign = self.softsign(embeddings)
        a = a + softsign.permute((0, 2, 1))
        x = a * torch.sigmoid(b)
        x = x + residual
        x = x * self.sqrt
        return x.permute((0, 2, 1))
