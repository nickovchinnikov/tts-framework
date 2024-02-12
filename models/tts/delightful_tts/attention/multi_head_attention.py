import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class MultiHeadAttention(Module):
    r"""A class that implements a Multi-head Attention mechanism.
    Multi-head attention allows the model to focus on different positions,
    capturing various aspects of the input.

    Args:
        query_dim (int): The dimensionality of the query.
        key_dim (int): The dimensionality of the key.
        num_units (int): The total number of dimensions of the output.
        num_heads (int): The number of parallel attention layers (multi-heads).

    Inputs: query, and key
        - **query**: Tensor of shape [N, T_q, query_dim]
        - **key**: Tensor of shape [N, T_k, key_dim]

    Outputs:
        - An output tensor of shape [N, T_q, num_units]
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

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        r"""Performs the forward pass over input tensors.

        Args:
            query (torch.Tensor): The input tensor containing query vectors.
                It is expected to have the dimensions [N, T_q, query_dim]
                where N is the batch size, T_q is the sequence length of queries,
                and query_dim is the dimensionality of a single query vector.

            key (torch.Tensor): The input tensor containing key vectors.
                It is expected to have the dimensions [N, T_k, key_dim]
                where N is the batch size, T_k is the sequence length of keys,
                and key_dim is the dimensionality of a single key vector.

        Returns:
            torch.Tensor: The output tensor of shape [N, T_q, num_units] which
                represents the results of the multi-head attention mechanism applied
                on the provided queries and keys.
        """
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)
        split_size = self.num_units // self.num_heads

        querys = torch.stack(
            torch.split(querys, split_size, dim=2), dim=0,
        )  # [h, N, T_q, num_units/h]
        keys = torch.stack(
            torch.split(keys, split_size, dim=2), dim=0,
        )  # [h, N, T_k, num_units/h]
        values = torch.stack(
            torch.split(values, split_size, dim=2), dim=0,
        )  # [h, N, T_k, num_units/h]
        # score = softmax(QK^T / (d_k ** 0.5))

        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim**0.5)
        scores = F.softmax(scores, dim=3)
        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        return torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(
            0,
        )  # [N, T_q, num_units]
