from math import log, pi

from einops import rearrange, repeat
import torch
from torch import Tensor, nn


class SinusoidalEmbedding(nn.Module):
    r"""Sinusoidal Embedding class that creates a sinusoidal embedding of a given dimension."""

    def __init__(self, dim: int):
        r"""Initialize the SinusoidalEmbedding with a dimension.

        Args:
            dim (int): The dimension of the embedding.
        """
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the SinusoidalEmbedding.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        device, half_dim = x.device, self.dim // 2

        emb = torch.tensor(log(10000) / (half_dim - 1), device=device)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = rearrange(x, "i -> i 1") * rearrange(emb, "j -> 1 j")

        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class LearnedPositionalEmbedding(nn.Module):
    r"""Learned Positional Embedding class that creates a learned positional embedding of a given dimension. Used for continuous time."""

    def __init__(self, dim: int):
        r"""Initialize the LearnedPositionalEmbedding with a dimension.

        Args:
            dim (int): The dimension of the embedding.
        """
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the LearnedPositionalEmbedding.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        x = rearrange(x, "b -> b 1")

        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi

        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)

        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    r"""Creates a time positional embedding of a given dimension and output features.

    Args:
        dim (int): The dimension of the embedding.
        out_features (int): The number of output features.

    Returns:
        nn.Module: The time positional embedding module.
    """
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
    )


class FixedEmbedding(nn.Module):
    r"""Fixed Embedding class that creates a fixed embedding of a given maximum length and features."""

    def __init__(self, max_length: int, features: int):
        r"""Initialize the FixedEmbedding with a maximum length and features.

        Args:
            max_length (int): The maximum length of the embedding.
            features (int): The number of features of the embedding.
        """
        super().__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(max_length, features)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the FixedEmbedding.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """
        batch_size, length, device = *x.shape[0:2], x.device

        assert length <= self.max_length, "Input sequence length must be <= max_length"

        position = torch.arange(length, device=device)

        fixed_embedding = self.embedding(position)
        fixed_embedding = repeat(fixed_embedding, "n d -> b n d", b=batch_size)

        return fixed_embedding
