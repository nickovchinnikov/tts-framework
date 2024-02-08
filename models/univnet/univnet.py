import torch
from torch import nn
from torch.nn import Module
from torch.nn.utils import parametrize

from models.config import PreprocessingConfig, VocoderModelConfig
from models.helpers.tools import get_mask_from_lengths

from .lvc_block import LVCBlock


class UnivNet(Module):
    """UnivNet Generator"""

    def __init__(
        self,
        model_config: VocoderModelConfig,
        preprocess_config: PreprocessingConfig,
    ):
        r"""UnivNet Generator.
        Initializes the UnivNet module.

        Args:
            model_config (VocoderModelConfig): the model configuration.
            preprocess_config (PreprocessingConfig): the preprocessing configuration.
        """
        super().__init__()

        self.mel_channel = preprocess_config.stft.n_mel_channels
        self.noise_dim = model_config.gen.noise_dim
        self.hop_length = preprocess_config.stft.hop_length
        channel_size = model_config.gen.channel_size
        kpnet_conv_size = model_config.gen.kpnet_conv_size

        hop_length = 1
        self.res_stack = nn.ModuleList()

        for stride in model_config.gen.strides:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    preprocess_config.stft.n_mel_channels,
                    stride=stride,
                    dilations=model_config.gen.dilations,
                    lReLU_slope=model_config.gen.lReLU_slope,
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size,
                ),
            )

        self.conv_pre = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                model_config.gen.noise_dim,
                channel_size,
                7,
                padding=3,
                padding_mode="reflect",
            ),
        )

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(model_config.gen.lReLU_slope),
            nn.utils.parametrizations.weight_norm(
                nn.Conv1d(
                    channel_size,
                    1,
                    7,
                    padding=3,
                    padding_mode="reflect",
                ),
            ),
            nn.Tanh(),
        )

        # Output of STFT(zeros)
        self.mel_mask_value = -11.5129

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the Generator module.

        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)

        Returns:
            Tensor: the generated audio waveform (batch, 1, out_length)
        """
        z = torch.randn(c.shape[0], self.noise_dim, c.shape[2], device=c.device)
        z = self.conv_pre(z)  # (B, c_g, L)

        for res_block in self.res_stack:
            z = res_block(z, c)  # (B, c_g, L * s_0 * ... * s_i)

        return self.conv_post(z)  # (B, 1, L * 256)

    def eval(self, inference: bool = False):
        r"""Sets the module to evaluation mode.

        Args:
            inference (bool): whether to remove weight normalization or not.
        """
        super().eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self) -> None:
        r"""Removes weight normalization from the module."""
        print("Removing weight norm...")

        parametrize.remove_parametrizations(self.conv_pre, "weight")

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                parametrize.remove_parametrizations(layer, "weight")

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def infer(self, c: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        r"""Infers the audio waveform from the mel-spectrogram conditioning sequence.

        Args:
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length)
            mel_lens (Tensor): the lengths of the mel-spectrogram conditioning sequence.

        Returns:
            Tensor: the generated audio waveform (batch, 1, out_length)
        """
        mel_mask = get_mask_from_lengths(mel_lens).unsqueeze(1).to(c.device)
        c = c.masked_fill(mel_mask, self.mel_mask_value)
        zero = torch.full(
            (c.shape[0], self.mel_channel, 10), self.mel_mask_value, device=c.device,
        )
        mel = torch.cat((c, zero), dim=2)
        audio = self(mel)
        audio = audio[:, :, : -(self.hop_length * 10)]
        audio_mask = get_mask_from_lengths(mel_lens * 256).unsqueeze(1)
        return audio.masked_fill(audio_mask, 0.0)
