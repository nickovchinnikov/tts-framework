import math
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers import ConvNorm, DiffusionEmbedding, LinearNorm, Mish, ResidualBlock


class Denoiser(nn.Module):
    r"""Conditional Diffusion Denoiser.

    This module implements a denoising model conditioned on a diffusion step, a conditioner, and a speaker embedding.
    It consists of several convolutional and linear projections followed by residual blocks.

    Args:
        preprocess_config (dict): Preprocessing configuration dictionary.
        model_config (dict): Model configuration dictionary.

    Attributes:
        input_projection (nn.Sequential): Sequential module for input projection.
        diffusion_embedding (DiffusionEmbedding): Diffusion step embedding module.
        mlp (nn.Sequential): Multilayer perceptron module.
        residual_layers (nn.ModuleList): List of residual blocks.
        skip_projection (ConvNorm): Convolutional projection for skip connections.
        output_projection (ConvNorm): Convolutional projection for output.

    """

    def __init__(self, preprocess_config, model_config):
        super().__init__()
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        d_encoder = model_config["transformer"]["encoder_hidden"]
        d_spk_prj = model_config["transformer"]["speaker_embed_dim"]
        residual_channels = model_config["denoiser"]["residual_channels"]
        residual_layers = model_config["denoiser"]["residual_layers"]
        dropout = model_config["denoiser"]["denoiser_dropout"]
        multi_speaker = model_config["multi_speaker"]

        self.input_projection = nn.Sequential(
            ConvNorm(n_mel_channels, residual_channels, kernel_size=1),
            nn.ReLU(),
        )
        self.diffusion_embedding = DiffusionEmbedding(residual_channels)
        self.mlp = nn.Sequential(
            LinearNorm(residual_channels, residual_channels * 4),
            Mish(),
            LinearNorm(residual_channels * 4, residual_channels),
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    d_encoder, residual_channels, dropout=dropout, d_spk_prj=d_spk_prj, multi_speaker=multi_speaker,
                )
                for _ in range(residual_layers)
            ],
        )
        self.skip_projection = ConvNorm(
            residual_channels, residual_channels, kernel_size=1,
        )
        self.output_projection = ConvNorm(
            residual_channels, n_mel_channels, kernel_size=1,
        )
        nn.init.zeros_(self.output_projection.conv.weight)

    def forward(
        self,
        mel: Tensor,
        diffusion_step: Tensor,
        conditioner: Tensor,
        speaker_emb: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Forward pass through the Denoiser module.

        Args:
            mel (torch.Tensor): Mel-spectrogram tensor of shape [B, 1, M, T].
            diffusion_step (torch.Tensor): Diffusion step tensor of shape [B,].
            conditioner (torch.Tensor): Conditioner tensor of shape [B, M, T].
            speaker_emb (torch.Tensor): Speaker embedding tensor of shape [B, M].
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.

        Returns:
            torch.Tensor: Output mel-spectrogram tensor of shape [B, 1, M, T].
        """
        x = mel[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, conditioner, diffusion_step, speaker_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]

        return x[:, None, :, :]
