import math

import torch
from torch import nn
import torch.nn.functional as F


class ConvNorm(nn.Module):
    """1D Convolution"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
    ):
        super(ConvNorm, self).__init__()

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

    def forward(self, signal):
        conv_signal = self.conv(signal)

        return conv_signal


class DiffusionEmbedding(nn.Module):
    """Diffusion Step Embedding"""

    def __init__(self, d_denoiser):
        super(DiffusionEmbedding, self).__init__()
        self.dim = d_denoiser

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearNorm(nn.Module):
    """LinearNorm Projection"""

    def __init__(self, in_features, out_features, bias=False):
        super(LinearNorm, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x):
        x = self.linear(x)
        return x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class ResidualBlock(nn.Module):
    """Residual Block"""

    def __init__(self, d_encoder, residual_channels, dropout, d_spk_prj, multi_speaker=True):
        super(ResidualBlock, self).__init__()
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

    def forward(self, x, conditioner, diffusion_step, speaker_emb, mask=None):

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


class Denoiser(nn.Module):
    """Conditional Diffusion Denoiser"""

    def __init__(self, preprocess_config, model_config):
        super(Denoiser, self).__init__()
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

    def forward(self, mel, diffusion_step, conditioner, speaker_emb, mask=None):
        """:param mel: [B, 1, M, T]
        :param diffusion_step: [B,]
        :param conditioner: [B, M, T]
        :param speaker_emb: [B, M]
        :return:
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
