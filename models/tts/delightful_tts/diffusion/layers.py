import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import Module
from torch.nn import functional as F


class Mish(Module):
    r"""Applies the Mish activation function.

    Mish is a smooth, non-monotonic function that attempts to mitigate the
    problems of dying ReLU units in deep neural networks.
    """

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass of the Mish activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying Mish activation.
        """
        return x * torch.tanh(F.softplus(x))


class ConvNorm(Module):
    r"""1D Convolution with optional batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolving kernel. Defaults to 1.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Zero-padding added to both sides of the input. Defaults to None.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Optional[int] = None,
        dilation: int = 1,
        bias: bool = True,
    ):
        super().__init__()

        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, signal: Tensor) -> Tensor:
        r"""Forward pass through the convolutional layer.

        Args:
            signal (torch.Tensor): Input signal tensor.

        Returns:
            torch.Tensor: Output tensor after convolution.
        """
        conv_signal = self.conv(signal)

        return conv_signal


class DiffusionEmbedding(Module):
    r"""Diffusion Step Embedding.

    This module generates diffusion step embeddings for the given input.

    Args:
        d_denoiser (int): Dimension of the denoiser.

    Attributes:
        dim (int): Dimension of the diffusion step embedding.
    """

    def __init__(self, d_denoiser: int):
        super().__init__()
        self.dim = d_denoiser

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass through the DiffusionEmbedding module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Diffusion step embeddings.
        """
        device = x.device
        half_dim = self.dim // 2

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class LinearNorm(Module):
    r"""LinearNorm Projection.

    This module performs a linear projection with optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): If True, adds a learnable bias to the output. Default is False.

    Attributes:
        linear (torch.nn.Linear): Linear transformation module.

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: Tensor) -> Tensor:
        r"""Forward pass through the LinearNorm module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after linear projection.
        """
        x = self.linear(x)
        return x


class ResidualBlock(Module):
    r"""Residual Block.

    This module defines a residual block used in a neural network architecture. It consists of
    several convolutional and linear projections followed by nonlinear activations.

    Args:
        d_encoder (int): Dimension of the encoder output.
        residual_channels (int): Number of channels in the residual block.
        dropout (float): Dropout probability.
        d_spk_prj (int): Dimension of the speaker projection.
        multi_speaker (bool, optional): Flag indicating if the model is trained with multiple speakers. Defaults to True.

    Attributes:
        multi_speaker (bool): Flag indicating if the model is trained with multiple speakers.
        conv_layer (ConvNorm): Convolutional layer in the residual block.
        diffusion_projection (LinearNorm): Linear projection for the diffusion step.
        speaker_projection (LinearNorm): Linear projection for the speaker embedding.
        conditioner_projection (ConvNorm): Convolutional projection for the conditioner.
        output_projection (ConvNorm): Convolutional projection for the output.
    """

    def __init__(
        self,
        d_encoder: int,
        residual_channels: int,
        dropout: float,
        d_spk_prj: int,
        multi_speaker: bool = True,
    ):
        super().__init__()
        self.multi_speaker = multi_speaker
        self.conv_layer = ConvNorm(
            residual_channels,
            2 * residual_channels,
            kernel_size=3,
            stride=1,
            padding=int((3 - 1) / 2),
            dilation=1,
        )
        self.diffusion_projection = LinearNorm(residual_channels, residual_channels)
        if multi_speaker:
            self.speaker_projection = LinearNorm(d_spk_prj, residual_channels)
        self.conditioner_projection = ConvNorm(
            d_encoder, residual_channels, kernel_size=1,
        )
        self.output_projection = ConvNorm(
            residual_channels, 2 * residual_channels, kernel_size=1,
        )

    def forward(
        self,
        x: Tensor,
        conditioner: Tensor,
        diffusion_step: Tensor,
        speaker_emb: Tensor,
        mask: Optional[Tensor] = None,
    ):
        r"""Forward pass through the ResidualBlock module.

        Args:
            x (torch.Tensor): Input tensor.
            conditioner (torch.Tensor): Conditioner tensor.
            diffusion_step (torch.Tensor): Diffusion step tensor.
            speaker_emb (torch.Tensor): Speaker embedding tensor.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the output tensor and skip tensor.
        """
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        # conditioner = self.conditioner_projection(conditioner)
        conditioner = self.conditioner_projection(conditioner.transpose(1, 2))
        if self.multi_speaker:
            # speaker_emb = self.speaker_projection(speaker_emb).unsqueeze(1).expand(
            #     -1, conditioner.shape[-1], -1,
            # ).transpose(1, 2)
            speaker_emb = self.speaker_projection(speaker_emb).expand(
                -1, conditioner.shape[-1], -1,
            ).transpose(1, 2)

        residual = y = x + diffusion_step
        y = self.conv_layer(
            (y + conditioner + speaker_emb) if self.multi_speaker else (y + conditioner),
        )
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        x, skip = torch.chunk(y, 2, dim=1)

        return (x + residual) / math.sqrt(2.0), skip
