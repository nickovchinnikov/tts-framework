import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class StyleEmbedAttention(Module):
    r"""Mechanism is being used to extract style features from audio data in the form of spectrograms.

    Each style token (parameterized by an embedding vector) represents a unique style feature. The model applies the `StyleEmbedAttention` mechanism to combine these style tokens (style features) in a weighted manner. The output of the attention module is a sum of style tokens, with each token weighted by its relevance to the input.

    This technique is often used in text-to-speech synthesis (TTS) such as Tacotron-2, where the goal is to modulate the prosody, stress, and intonation of the synthesized speech based on the reference audio or some control parameters. The concept of "global style tokens" (GST) was introduced in
    [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017) by Yuxuan Wang et al.

    The `StyleEmbedAttention` class is a PyTorch module implementing the attention mechanism.
    This class is specifically designed for handling multiple attention heads.
    Attention here operates on a query and a set of key-value pairs to produce an output.

    Builds the `StyleEmbedAttention` network.

    Args:
        query_dim (int): Dimensionality of the query vectors.
        key_dim (int): Dimensionality of the key vectors.
        num_units (int): Total dimensionality of the query, key, and value vectors.
        num_heads (int): Number of parallel attention layers (heads).

    Note: `num_units` should be divisible by `num_heads`.
    """

    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        num_units: int,
        num_heads: int,
    ):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(
            in_features=query_dim,
            out_features=num_units,
            bias=False,
        )
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(
            in_features=key_dim, out_features=num_units, bias=False,
        )

    def forward(self, query: torch.Tensor, key_soft: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the StyleEmbedAttention module calculates the attention scores.

        Args:
            query (torch.Tensor): The input tensor for queries of shape `[N, T_q, query_dim]`
            key_soft (torch.Tensor): The input tensor for keys of shape `[N, T_k, key_dim]`

        Returns:
            out (torch.Tensor): The output tensor of shape `[N, T_q, num_units]`
        """
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        # out_soft = scores_soft = None
        queries = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key_soft)  # [N, T_k, num_units]

        # [h, N, T_q, num_units/h]
        queries = torch.stack(torch.split(queries, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores_soft = torch.matmul(queries, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores_soft = scores_soft / (self.key_dim**0.5)
        scores_soft = F.softmax(scores_soft, dim=3)

        # out = score * V
        # [h, N, T_q, num_units/h]
        out_soft = torch.matmul(scores_soft, values)
        return torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(
            0,
        )  # [N, T_q, num_units] scores_soft
