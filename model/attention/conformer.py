import torch
import torch.nn as nn

from model.attention import ConformerBlock
from model.basenn import BaseNNModule

from model.helpers.tools import get_device


class Conformer(BaseNNModule):
    r"""
    `Conformer` class represents the `Conformer` model which is a sequence-to-sequence model
    used in some modern automated speech recognition systems. It is composed of several `ConformerBlocks`.

    Args:
        dim (int): The number of expected features in the input.
        n_layers (int): The number of `ConformerBlocks` in the Conformer model.
        n_heads (int): The number of heads in the multiheaded self-attention mechanism in each `ConformerBlock`.
        embedding_dim (int): The dimension of the embeddings.
        p_dropout (float): The dropout probability to be used in each `ConformerBlock`.
        kernel_size_conv_mod (int): The size of the convolving kernel in the convolution module of each `ConformerBlock`.
        with_ff (bool): If True, each `ConformerBlock` uses FeedForward layer inside it.
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`
    """

    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        embedding_dim: int,
        p_dropout: float,
        kernel_size_conv_mod: int,
        with_ff: bool,
        device: torch.device = get_device(),
    ):
        super().__init__(device)
        d_k = d_v = dim // n_heads
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    n_heads,
                    d_k,
                    d_v,
                    kernel_size_conv_mod=kernel_size_conv_mod,
                    dropout=p_dropout,
                    embedding_dim=embedding_dim,
                    with_ff=with_ff,
                    device=self.device,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        embeddings: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        Forward Pass of the Conformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            mask (Tensor): The mask tensor.
            embeddings (Tensor): Embeddings tensor.
            encoding (Tensor): The positional encoding tensor.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, num_features).
        """
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                mask=mask,
                slf_attn_mask=attn_mask,
                embeddings=embeddings,
                encoding=encoding,
            )
        return x
