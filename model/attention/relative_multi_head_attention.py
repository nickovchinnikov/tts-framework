import math
from typing import Tuple

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F


class RelativeMultiHeadAttention(Module):
    r"""Multi-head attention with relative positional encoding.
    This concept was proposed in the
    [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860)

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.

    Inputs: query, key, value, pos_embedding, mask
        - **query** (batch, time, dim): Tensor containing query vector
        - **key** (batch, time, dim): Tensor containing key vector
        - **value** (batch, time, dim): Tensor containing value vector
        - **pos_embedding** (batch, time, dim): Positional embedding tensor
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked
    Returns:
        - **outputs**: Tensor produces by relative multi head attention module.

    Note: `d_model` should be divisible by `num_heads` in other words `d_model % num_heads` should be zero.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 16,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model % num_heads should be zero."
        self.d_model = d_model
        self.d_head = int(d_model / num_heads)
        self.num_heads = num_heads
        self.sqrt_dim = math.sqrt(d_model)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_proj = nn.Linear(d_model, d_model, bias=False)

        self.u_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))
        self.v_bias = nn.Parameter(torch.Tensor(self.num_heads, self.d_head))

        torch.nn.init.xavier_uniform_(self.u_bias)
        torch.nn.init.xavier_uniform_(self.v_bias)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        pos_embedding: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Function applies multi-head attention along with relative positional encoding to the inputs. It restructures the input queries, keys, and values according to individual attention heads, applies biases, calculates content and position scores, and combines these to get the final score. A softmax activation is applied over the final score, followed by the calculation of context (contextual representation of input).

        Performs the forward pass on the queries, keys, values, and positional embeddings with a mask.

        Args:
            query (torch.Tensor): The input tensor containing query vectors.
            key (torch.Tensor): The input tensor containing key vectors.
            value (torch.Tensor): The input tensor containing value vectors.
            pos_embedding (torch.Tensor): The positional embedding tensor.
            mask (torch.Tensor): The mask tensor containing indices to be masked.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The context and attention tensors.
            Tensor produces by relative multi head attention module.
        """
        batch_size = query.shape[0]
        query = self.query_proj(query).view(batch_size, -1, self.num_heads, self.d_head)
        key = (
            self.key_proj(key)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.d_head)
            .permute(0, 2, 1, 3)
        )
        pos_embedding = self.pos_proj(pos_embedding).view(
            batch_size, -1, self.num_heads, self.d_head,
        )
        u_bias = self.u_bias.expand_as(query)
        v_bias = self.v_bias.expand_as(query)
        a = (query + u_bias).transpose(1, 2)
        content_score = a @ key.transpose(2, 3)
        b = (query + v_bias).transpose(1, 2)
        pos_score = b @ pos_embedding.permute(0, 2, 3, 1)
        pos_score = self._relative_shift(pos_score)

        score = content_score + pos_score
        score = score * (1.0 / self.sqrt_dim)

        score.masked_fill_(mask, -1e9)

        attn = F.softmax(score, -1)

        context = (attn @ value).transpose(1, 2)
        context = context.contiguous().view(batch_size, -1, self.d_model)

        return self.out_proj(context), attn

    def _relative_shift(self, pos_score: torch.Tensor) -> torch.Tensor:
        r"""The main idea of relative positional encoding is that the attention score doesn't only depend on the query and the key, but also on the relative position of the key with respect to the query. This becomes particularly useful when working with sequences of tokens, like in NLP tasks, as it helps the model to be aware of the position of the words (or tokens) in the sentence.

        Performs the relative shift operation on the positional scores.

        Args:
            pos_score (torch.Tensor): The positional scores tensor.

        Returns:
            torch.Tensor: The shifted positional scores tensor.
        """
        batch_size, num_heads, seq_length1, seq_length2 = pos_score.size()
        zeros = torch.zeros(
            (batch_size, num_heads, seq_length1, 1), device=pos_score.device,
        )
        padded_pos_score = torch.cat([zeros, pos_score], dim=-1)
        padded_pos_score = padded_pos_score.view(
            batch_size, num_heads, seq_length2 + 1, seq_length1,
        )
        return padded_pos_score[:, :, 1:].view_as(pos_score)
