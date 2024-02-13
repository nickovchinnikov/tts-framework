from typing import Optional

from einops import reduce as einops_reduce
from einops.layers.torch import Rearrange
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .attention import AttentionBase, FeedForward
from .embeddings import FixedEmbedding, TimePositionalEmbedding
from .utils import default, exists, rand_bool


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

    def forward(self, x: Tensor, s: Tensor):
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


class StyleTransformer1d(nn.Module):
    def __init__(
        self,
        num_layers: int,
        channels: int,
        num_heads: int,
        head_features: int,
        multiplier: int,
        context_embedding_features: int,
        context_features: int,
        use_context_time: bool = True,
        use_rel_pos: bool = False,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        embedding_max_length: int = 512,
    ):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                StyleTransformerBlock(
                    features=channels + context_embedding_features,
                    head_features=head_features,
                    num_heads=num_heads,
                    multiplier=multiplier,
                    style_dim=context_features,
                    use_rel_pos=use_rel_pos,
                    rel_pos_num_buckets=rel_pos_num_buckets,
                    rel_pos_max_distance=rel_pos_max_distance,
                )
                for i in range(num_layers)
            ],
        )

        self.to_out = nn.Sequential(
            Rearrange("b t c -> b c t"),
            nn.Conv1d(
                in_channels=channels + context_embedding_features,
                out_channels=channels,
                kernel_size=1,
            ),
        )

        use_context_features = exists(context_features)
        self.use_context_features = use_context_features
        self.use_context_time = use_context_time

        context_mapping_features = channels + context_embedding_features

        if use_context_time or use_context_features:
            self.to_mapping = nn.Sequential(
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
                nn.Linear(context_mapping_features, context_mapping_features),
                nn.GELU(),
            )

        if use_context_time:
            assert exists(context_mapping_features)
            self.to_time = nn.Sequential(
                TimePositionalEmbedding(
                    dim=channels, out_features=context_mapping_features,
                ),
                nn.GELU(),
            )

        if use_context_features:
            assert exists(context_features) and exists(context_mapping_features)
            self.to_features = nn.Sequential(
                nn.Linear(
                    in_features=context_features, out_features=context_mapping_features,
                ),
                nn.GELU(),
            )

        self.fixed_embedding = FixedEmbedding(
            max_length=embedding_max_length, features=context_embedding_features,
        )

    def get_mapping(
        self, time: Optional[Tensor] = None, features: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """Combines context time features and features into mapping"""
        items, mapping = [], None

        # Compute time features
        if self.use_context_time:
            assert_message = "use_context_time=True but no time features provided"
            assert exists(time), assert_message
            items += [self.to_time(time)]

        # Compute features
        if self.use_context_features:
            assert_message = "context_features exists but no features provided"
            assert exists(features), assert_message
            items += [self.to_features(features)]

        # Compute joint mapping
        if self.use_context_time or self.use_context_features:
            mapping = einops_reduce(torch.stack(items), "n b m -> b m", "sum")
            mapping = self.to_mapping(mapping)

        return mapping

    def run(self, x: Tensor, time: Tensor, embedding: Tensor, features: Optional[Tensor]) -> Tensor:
        mapping = self.get_mapping(time, features)
        x = torch.cat((x.expand(-1, embedding.size(1), -1), embedding), dim=-1)

        if mapping is not None:
            mapping = mapping.unsqueeze(1).expand(-1, embedding.size(1), -1)

        for block in self.blocks:
            x = x + mapping
            x = block(x, features)

        x = x.mean(dim=1).unsqueeze(1)
        x = self.to_out(x)
        x = x.transpose(-1, -2)

        return x

    def forward(self, x: Tensor,
                time: Tensor,
                embedding_mask_proba: float = 0.0,
                embedding: Tensor = torch.tensor([]),
                features: Optional[Tensor] = None,
                embedding_scale: float = 1.0) -> Tensor:

        b, device = embedding.shape[0], embedding.device
        self.fixed_embedding = self.fixed_embedding.to(device)

        fixed_embedding = self.fixed_embedding(embedding)
        if embedding_mask_proba > 0.0:
            # Randomly mask embedding
            batch_mask = rand_bool(
                shape=(b, 1, 1), proba=embedding_mask_proba,
            ).to(device)

            embedding = torch.where(batch_mask, fixed_embedding, embedding)

        if embedding_scale != 1.0:
            # Compute both normal and fixed embedding outputs
            out = self.run(x, time, embedding=embedding, features=features)
            out_masked = self.run(x, time, embedding=fixed_embedding, features=features)
            # Scale conditional output using classifier-free guidance
            return out_masked + (out - out_masked) * embedding_scale
        else:
            return self.run(x, time, embedding=embedding, features=features)


class StyleTransformerBlock(nn.Module):
    def __init__(
        self,
        features: int,
        num_heads: int,
        head_features: int,
        style_dim: int,
        multiplier: int,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
        context_features: Optional[int] = None,
    ):
        super().__init__()

        self.use_cross_attention = exists(context_features) and context_features > 0

        self.attention = StyleAttention(
            features=features,
            style_dim=style_dim,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

        if self.use_cross_attention:
            self.cross_attention = StyleAttention(
                features=features,
                style_dim=style_dim,
                num_heads=num_heads,
                head_features=head_features,
                context_features=context_features,
                use_rel_pos=use_rel_pos,
                rel_pos_num_buckets=rel_pos_num_buckets,
                rel_pos_max_distance=rel_pos_max_distance,
            )

        self.feed_forward = FeedForward(features=features, multiplier=multiplier)

    def forward(self, x: Tensor, s: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        x = self.attention(x, s) + x
        if self.use_cross_attention:
            x = self.cross_attention(x, s, context=context) + x
        x = self.feed_forward(x) + x
        return x


class StyleAttention(nn.Module):
    def __init__(
        self,
        features: int,
        *,
        style_dim: int,
        head_features: int,
        num_heads: int,
        context_features: Optional[int] = None,
        use_rel_pos: bool,
        rel_pos_num_buckets: Optional[int] = None,
        rel_pos_max_distance: Optional[int] = None,
    ):
        super().__init__()
        self.context_features = context_features
        mid_features = head_features * num_heads
        context_features = default(context_features, features)

        self.norm = AdaLayerNorm(style_dim, features)
        self.norm_context = AdaLayerNorm(style_dim, context_features)
        self.to_q = nn.Linear(
            in_features=features, out_features=mid_features, bias=False,
        )
        self.to_kv = nn.Linear(
            in_features=context_features, out_features=mid_features * 2, bias=False,
        )
        self.attention = AttentionBase(
            features,
            num_heads=num_heads,
            head_features=head_features,
            use_rel_pos=use_rel_pos,
            rel_pos_num_buckets=rel_pos_num_buckets,
            rel_pos_max_distance=rel_pos_max_distance,
        )

    def forward(self, x: Tensor, s: Tensor, *, context: Optional[Tensor] = None) -> Tensor:
        assert_message = "You must provide a context when using context_features"
        assert not self.context_features or exists(context), assert_message
        # Use context if provided
        context = default(context, x)
        # Normalize then compute q from input and k,v from context
        x, context = self.norm(x, s), self.norm_context(context, s)

        q, k, v = (self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1))
        # Compute and return attention
        return self.attention(q, k, v)
