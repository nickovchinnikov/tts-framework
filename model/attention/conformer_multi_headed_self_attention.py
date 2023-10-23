from typing import Tuple

import torch
from torch import nn
from torch.nn import Module

from .relative_multi_head_attention import RelativeMultiHeadAttention


class ConformerMultiHeadedSelfAttention(Module):
    """Conformer employ multi-headed self-attention (MHSA) while integrating an important technique from Transformer-XL,
    the relative sinusoidal positional encoding scheme. The relative positional encoding allows the self-attention
    module to generalize better on different input length and the resulting encoder is more robust to the variance of
    the utterance length. Conformer use `prenorm` residual units with dropout which helps training
    and regularizing deeper models.

    Args:
        d_model (int): The dimension of model
        num_heads (int): The number of attention heads.
        dropout_p (float): probability of dropout

    Inputs: inputs, mask
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **mask** (batch, 1, time2) or (batch, time1, time2): Tensor containing indices to be masked

    Returns:
        (batch, time, dim): Tensor produces by relative multi headed self attention module.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float,
    ):
        super().__init__()

        # Initialize the RelativeMultiHeadAttention module passing the model dimension and number of attention heads
        self.attention = RelativeMultiHeadAttention(
            d_model=d_model, num_heads=num_heads,
        )
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, _, _ = key.size()

        # Trim or extend the "encoding" to match the size of key, and repeat this for each input in the batch
        encoding = encoding[:, : key.shape[1]]
        encoding = encoding.repeat(batch_size, 1, 1)

        # Pass inputs through the RelativeMultiHeadAttention layer, dropout the resulting outputs
        outputs, attn = self.attention(
            query, key, value, pos_embedding=encoding, mask=mask,
        )

        # Apply dropout to the attention outputs
        outputs = self.dropout(outputs)
        return outputs, attn
