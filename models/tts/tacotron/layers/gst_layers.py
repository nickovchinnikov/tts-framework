from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class GST(nn.Module):
    r"""Global Style Token Module for factorizing prosody in speech.

    Args:
        num_mel (int): Number of mel filters.
        num_heads (int): Number of attention heads.
        num_style_tokens (int): Number of style tokens.
        gst_embedding_dim (int): Dimension of the GST embedding.
        embedded_speaker_dim (int, optional): Dimension of the embedded speaker. Defaults to None.

    Attributes:
        encoder (ReferenceEncoder): Reference encoder.
        style_token_layer (StyleTokenLayer): Style token layer.

    See https://arxiv.org/pdf/1803.09017
    """

    def __init__(
        self,
        num_mel: int,
        num_heads: int,
        num_style_tokens: int,
        gst_embedding_dim: int,
        embedded_speaker_dim: Optional[int]=None,
    ):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mel, gst_embedding_dim)
        self.style_token_layer = StyleTokenLayer(num_heads, num_style_tokens, gst_embedding_dim, embedded_speaker_dim)

    def forward(self, inputs: Tensor, speaker_embedding: Optional[Tensor]=None):
        r"""Forward pass of the GST.

        Args:
            inputs (torch.Tensor): Input tensor.
            speaker_embedding (Optional[torch.Tensor], optional): Speaker embedding tensor. Defaults to None.

        Returns:
            torch.Tensor: Style embedding tensor.
        """
        enc_out = self.encoder(inputs)
        # concat speaker_embedding
        if speaker_embedding is not None:
            enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)
        style_embed = self.style_token_layer(enc_out)

        return style_embed


class ReferenceEncoder(nn.Module):
    r"""NN module creating a fixed size prosody embedding from a spectrogram.

    Args:
        num_mel (int): Number of mel filters.
        embedding_dim (int): Dimension of the output embedding.

    Attributes:
        num_mel (int): Number of mel filters.
        convs (nn.ModuleList): List of convolutional layers.
        bns (nn.ModuleList): List of batch normalization layers.
        recurrence (nn.GRU): GRU layer.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel: int, embedding_dim: int):
        super().__init__()
        self.num_mel = num_mel
        filters = [1, 32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim // 2, batch_first=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Forward pass of the ReferenceEncoder.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_spec_frames, num_mel).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, embedding_dim).
        """
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        return out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(
        height: int,
        kernel_size: int,
        stride: int,
        pad: int,
        n_convs: int,
    ) -> int:
        r"""Calculate the height of the output after n convolutions.

        Args:
            height (int): Initial height.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            pad (int): Padding of the convolution.
            n_convs (int): Number of convolutions.

        Returns:
            int: The height of the output after n convolutions.
        """
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


class StyleTokenLayer(nn.Module):
    """NN Module attending to style tokens based on prosody encodings.

    Args:
        num_heads (int): Number of attention heads.
        num_style_tokens (int): Number of style tokens.
        gst_embedding_dim (int): Dimension of the GST embedding.
        d_vector_dim (int, optional): Dimension of the d vector. Defaults to None.

    Attributes:
        query_dim (int): Dimension of the query.
        key_dim (int): Dimension of the key.
        style_tokens (nn.Parameter): Style tokens.
        attention (MultiHeadAttention): Multi-head attention layer.
    """

    def __init__(
        self,
        num_heads: int,
        num_style_tokens: int,
        gst_embedding_dim: int,
        d_vector_dim: Optional[int]=None,
    ):
        super().__init__()

        self.query_dim = gst_embedding_dim // 2

        if d_vector_dim:
            self.query_dim += d_vector_dim

        self.key_dim = gst_embedding_dim // num_heads
        self.style_tokens = nn.Parameter(torch.FloatTensor(num_style_tokens, self.key_dim))
        nn.init.normal_(self.style_tokens, mean=0, std=0.5)
        self.attention = MultiHeadAttention(
            query_dim=self.query_dim, key_dim=self.key_dim, num_units=gst_embedding_dim, num_heads=num_heads,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        r"""Forward pass of the StyleTokenLayer.

        Args:
            inputs (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Style embedding tensor.
        """
        batch_size = inputs.size(0)
        prosody_encoding = inputs.unsqueeze(1)
        # prosody_encoding: 3D tensor [batch_size, 1, encoding_size==128]
        tokens = torch.tanh(self.style_tokens).unsqueeze(0).expand(batch_size, -1, -1)
        # tokens: 3D tensor [batch_size, num tokens, token embedding size]
        style_embed = self.attention(prosody_encoding, tokens)

        return style_embed


class MultiHeadAttention(nn.Module):
    r"""Multi-head attention layer.

    Args:
        query_dim (int): Dimension of the query.
        key_dim (int): Dimension of the key.
        num_units (int): Number of units.
        num_heads (int): Number of attention heads.

    Attributes:
        num_units (int): Number of units.
        num_heads (int): Number of attention heads.
        key_dim (int): Dimension of the key.
        W_query (nn.Linear): Linear layer for the query.
        W_key (nn.Linear): Linear layer for the key.
        W_value (nn.Linear): Linear layer for the value.

    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim: int, key_dim: int, num_units: int, num_heads: int):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query: Tensor, key: Tensor) -> Tensor:
        r"""Forward pass of the MultiHeadAttention.

        Args:
            query (torch.Tensor): Query tensor.
            key (torch.Tensor): Key tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        queries = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        queries = torch.stack(torch.split(queries, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k**0.5))
        scores = torch.matmul(queries, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim**0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
