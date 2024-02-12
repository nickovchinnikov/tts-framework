from math import log
from typing import Optional

from einops import rearrange
import torch
from torch import Tensor, einsum, nn

from models.tts.styledtts2.einops_exts import rearrange_many

from .utils import default, exists


class RelativePositionBias(nn.Module):
    r"""RelativePositionBias class that creates a relative position bias for attention mechanisms."""

    def __init__(self, num_buckets: int, max_distance: int, num_heads: int):
        r"""Initialize the RelativePositionBias with a number of buckets, maximum distance, and number of heads.

        Args:
            num_buckets (int): The number of buckets for the relative position bias.
            max_distance (int): The maximum distance for the relative position bias.
            num_heads (int): The number of heads for the relative position bias.
        """
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(
        relative_position: Tensor, num_buckets: int, max_distance: int,
    ) -> Tensor:
        r"""Compute the relative position bucket.

        Args:
            relative_position (Tensor): The relative position tensor.
            num_buckets (int): The number of buckets.
            max_distance (int): The maximum distance.

        Returns:
            Tensor: The relative position bucket tensor.
        """
        num_buckets //= 2
        ret = (relative_position >= 0).to(torch.long) * num_buckets
        n = torch.abs(relative_position)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(n.float() / max_exact)
                / log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1),
        )

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, num_queries: int, num_keys: int) -> Tensor:
        r"""Forward pass of the RelativePositionBias.

        Args:
            num_queries (int): The number of queries.
            num_keys (int): The number of keys.

        Returns:
            Tensor: The output tensor.
        """
        i, j, device = num_queries, num_keys, self.relative_attention_bias.weight.device
        q_pos = torch.arange(j - i, j, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")

        relative_position_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance,
        )

        bias = self.relative_attention_bias(relative_position_bucket)
        bias = rearrange(bias, "m n h -> 1 h m n")
        return bias


def FeedForward(features: int, multiplier: int) -> nn.Module:
    r"""Creates a feed-forward neural network with GELU activation in the middle layer.

    Args:
        features (int): The number of input and output features.
        multiplier (int): The factor to multiply the number of features to get the number of features in the middle layer.

    Returns:
        nn.Module: A feed-forward neural network module.
    """
    mid_features = features * multiplier
    return nn.Sequential(
        nn.Linear(in_features=features, out_features=mid_features),
        nn.GELU(),
        nn.Linear(in_features=mid_features, out_features=features),
    )


class AttentionBase(nn.Module):
    r"""AttentionBase class that creates a base attention mechanism."""

    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        use_rel_pos: bool,
        out_features: Optional[int] = None,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        r"""Initialize the AttentionBase with features, head features, number of heads, and relative position parameters.

        Args:
            features (int): The number of input features.
            head_features (int): The number of features in each head.
            num_heads (int): The number of heads.
            use_rel_pos (bool): Whether to use relative position bias.
            out_features (Optional[int]): The number of output features. If None, it will be set to the number of input features.
            rel_pos_num_buckets (Optional[int]): The number of buckets for relative position bias. Required if use_rel_pos is True.
            rel_pos_max_distance (Optional[int]): The maximum distance for relative position bias. Required if use_rel_pos is True.
        """
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        self.use_rel_pos = use_rel_pos
        mid_features = head_features * num_heads

        if use_rel_pos:
            if not exists(rel_pos_num_buckets):
                raise ValueError("rel_pos_num_buckets must be provided.")
            if not exists(rel_pos_max_distance):
                raise ValueError("rel_pos_max_distance must be provided.")

            self.rel_pos = RelativePositionBias(
                num_buckets=rel_pos_num_buckets,
                max_distance=rel_pos_max_distance,
                num_heads=num_heads,
            )
        if out_features is None:
            out_features = features

        self.to_out = nn.Linear(in_features=mid_features, out_features=out_features)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        r"""Forward pass of the AttentionBase.

        Args:
            q (Tensor): The query tensor.
            k (Tensor): The key tensor.
            v (Tensor): The value tensor.

        Returns:
            Tensor: The output tensor.
        """
        # Split heads
        q, k, v = rearrange_many((q, k, v), "b n (h d) -> b h n d", h=self.num_heads)
        # Compute similarity matrix
        sim = einsum("... n d, ... m d -> ... n m", q, k)
        sim = (sim + self.rel_pos(*sim.shape[-2:])) if self.use_rel_pos else sim
        sim = sim * self.scale
        # Get attention matrix with softmax
        attn = sim.softmax(dim=-1)
        # Compute values
        out = einsum("... n m, ... m d -> ... n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention(nn.Module):
    r"""Attention class that creates an attention mechanism with optional context."""

    def __init__(
        self,
        features: int,
        *,
        head_features: int,
        num_heads: int,
        out_features: Optional[int] = None,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        r"""Initialize the Attention with features, head features, number of heads, and relative position parameters.

        Args:
            features (int): The number of input features.
            head_features (int): The number of features in each head.
            num_heads (int): The number of heads.
            out_features (Optional[int]): The number of output features. If None, it will be set to the number of input features.
            context_features (Optional[int]): The number of context features. If None, it will be set to the number of input features.
            use_rel_pos (bool): Whether to use relative position bias.
            rel_pos_num_buckets (Optional[int]): The number of buckets for relative position bias. Required if use_rel_pos is True.
            rel_pos_max_distance (Optional[int]): The maximum distance for relative position bias. Required if use_rel_pos is True.
        """
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = nn.LayerNorm(features)
        self.norm_context = nn.LayerNorm(context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False,
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False,
        )

        self.attention = AttentionBase(
            features,
            out_features=out_features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        r"""Forward pass of the Attention.

        Args:
            x (Tensor): The input tensor.
            context (Optional[Tensor]): The context tensor. If None, the input tensor will be used as the context.

        Returns:
            Tensor: The output tensor.
        """
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message

        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x), self.norm_context(context)
        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))

        # Compute and return attention
        return self.attention(q, k, v)
