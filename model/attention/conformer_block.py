from lightning.pytorch import LightningModule
import torch
import torch.nn as nn

from model.conv_blocks import Conv1dGLU

from .conformer_conv_module import ConformerConvModule
from .conformer_multi_headed_self_attention import ConformerMultiHeadedSelfAttention
from .feed_forward import FeedForward


class ConformerBlock(LightningModule):
    r"""
    ConformerBlock class represents a block in the Conformer model architecture.
    The block includes a pointwise convolution followed by Gated Linear Units (`GLU`) activation layer (`Conv1dGLU`),
    a Conformer self attention layer (`ConformerMultiHeadedSelfAttention`), and optional feed-forward layer (`FeedForward`).

    Args:
        d_model (int): The number of expected features in the input.
        n_head (int): The number of heads for the multiheaded attention mechanism.
        d_k (int): The dimension of the key vectors for the attention mechanism.
        d_v (int): The dimension of the value vectors for the attention mechanism.
        kernel_size_conv_mod (int): The size of the convolving kernel for the convolution module.
        embedding_dim (int): The dimension of the embeddings.
        dropout (float): The dropout probability.
        with_ff (bool): If True, uses FeedForward layer inside ConformerBlock.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_k: int,
        d_v: int,
        kernel_size_conv_mod: int,
        embedding_dim: int,
        dropout: float,
        with_ff: bool,
    ):
        super().__init__()
        self.with_ff = with_ff
        self.conditioning = Conv1dGLU(
            d_model=d_model,
            kernel_size=kernel_size_conv_mod,
            padding=kernel_size_conv_mod // 2,
            embedding_dim=embedding_dim,
        )
        if self.with_ff:
            self.ff = FeedForward(
                d_model=d_model,
                dropout=dropout,
                kernel_size=3,
            )
        self.conformer_conv_1 = ConformerConvModule(
            d_model,
            kernel_size=kernel_size_conv_mod,
            dropout=dropout,
        )
        self.ln = nn.LayerNorm(
            d_model,
        )
        self.slf_attn = ConformerMultiHeadedSelfAttention(
            d_model=d_model,
            num_heads=n_head,
            dropout_p=dropout,
        )
        self.conformer_conv_2 = ConformerConvModule(
            d_model,
            kernel_size=kernel_size_conv_mod,
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
        slf_attn_mask: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward pass of the Conformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            embeddings (Tensor): Embeddings tensor.
            mask (Tensor): The mask tensor.
            slf_attn_mask (Tensor): The mask for self-attention layer.
            encoding (Tensor): The positional encoding tensor.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, num_features).
        """
        x = self.conditioning(x, embeddings=embeddings)
        if self.with_ff:
            x = self.ff(x) + x
        x = self.conformer_conv_1(x) + x
        res = x
        x = self.ln(x)
        x, _ = self.slf_attn(
            query=x, key=x, value=x, mask=slf_attn_mask, encoding=encoding
        )
        x = x + res
        x = x.masked_fill(mask.unsqueeze(-1), 0)
        x = self.conformer_conv_2(x) + x
        return x
