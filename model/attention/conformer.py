import torch
from torch import nn
from torch.nn import Module

from .conformer_block import ConformerBlock


class Conformer(Module):
    r"""`Conformer` class represents the `Conformer` model which is a sequence-to-sequence model
    used in some modern automated speech recognition systems. It is composed of several `ConformerBlocks`.

    Args:
        dim (int): The number of expected features in the input.
        n_layers (int): The number of `ConformerBlocks` in the Conformer model.
        n_heads (int): The number of heads in the multiheaded self-attention mechanism in each `ConformerBlock`.
        embedding_dim (int): The dimension of the embeddings.
        p_dropout (float): The dropout probability to be used in each `ConformerBlock`.
        kernel_size_conv_mod (int): The size of the convolving kernel in the convolution module of each `ConformerBlock`.
        with_ff (bool): If True, each `ConformerBlock` uses FeedForward layer inside it.
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
    ):
        super().__init__()
        self.layer_stack = nn.ModuleList(
            [
                ConformerBlock(
                    dim,
                    n_heads,
                    kernel_size_conv_mod=kernel_size_conv_mod,
                    dropout=p_dropout,
                    embedding_dim=embedding_dim,
                    with_ff=with_ff,
                )
                for _ in range(n_layers)
            ],
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        embeddings: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward Pass of the Conformer block.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            mask (Tensor): The mask tensor.
            embeddings (Tensor): Embeddings tensor.
            encoding (Tensor): The positional encoding tensor.

        Returns:
            Tensor: The output tensor of shape (batch_size, seq_len, num_features).
        """
        attn_mask = mask.view((mask.shape[0], 1, 1, mask.shape[1]))
        attn_mask.to(x.device)
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                mask=mask,
                slf_attn_mask=attn_mask,
                embeddings=embeddings,
                encoding=encoding,
            )
        return x
