from lightning.pytorch import LightningModule
import torch
import torch.nn as nn

from model.attention import StyleEmbedAttention
from model.config import AcousticModelConfigType


class STL(LightningModule):
    r"""
    Style Token Layer (STL).
    This layer helps to encapsulate different speaking styles in token embeddings.

    Args:
        model_config (AcousticModelConfigType): An object containing the model's configuration parameters.

    Attributes:
        embed (nn.Parameter): The style token embedding tensor.
        attention (StyleEmbedAttention): The attention module used to compute a weighted sum of embeddings.
    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()

        # Number of attention heads
        num_heads = 1
        # Dimension of encoder hidden states
        n_hidden = model_config.encoder.n_hidden
        # Number of style tokens
        self.token_num = model_config.reference_encoder.token_num

        # Define a learnable tensor for style tokens embedding
        self.embed = nn.Parameter(
            torch.FloatTensor(self.token_num, n_hidden // num_heads)
        )

        # Dimension of query in attention
        d_q = n_hidden // 2
        # Dimension of keys in attention
        d_k = n_hidden // num_heads

        # Style Embedding Attention module
        self.attention = StyleEmbedAttention(
            query_dim=d_q,
            key_dim=d_k,
            num_units=n_hidden,
            num_heads=num_heads,
        )

        # Initialize the embedding with normal distribution
        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Forward pass of the Style Token Layer
        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The emotion embedded tensor after applying attention mechanism.
        """
        N = x.size(0)

        # Reshape input tensor to [N, 1, n_hidden // 2]
        query = x.unsqueeze(1)

        keys_soft = (
            torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)
        )  # [N, token_num, n_hidden // num_heads]

        # Apply attention mechanism to get weighted sum of style token embeddings
        emotion_embed_soft = self.attention(query, keys_soft)

        return emotion_embed_soft
