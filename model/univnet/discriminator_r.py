from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from helpers.tools import get_device

from model.basenn import BaseNNModule

from torch.nn.utils import spectral_norm, weight_norm

from config import VocoderModelConfig


class DiscriminatorR(BaseNNModule):
    r"""
    A class representing the Residual Discriminator network for a UnivNet vocoder.

    Args:
        resolution (tuple): A tuple containing the number of FFT points, hop length, and window length.
        model_config (VocoderModelConfig): A configuration object for the UnivNet model.
        device (torch.device, optional): The device to use for the model. Defaults to the result of `get_device()`.
    """

    def __init__(
        self,
        resolution,
        model_config: VocoderModelConfig,
        device: torch.device = get_device(),
    ):
        super(DiscriminatorR, self).__init__(device=device)

        self.resolution = resolution
        self.LRELU_SLOPE = model_config.mrd.lReLU_slope

        # Use spectral normalization or weight normalization based on the configuration
        norm_f: Any = (
            spectral_norm if model_config.mrd.use_spectral_norm else weight_norm
        )

        # Define the convolutional layers
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv2d(1, 32, (3, 9), padding=(1, 4), device=self.device)),
                norm_f(
                    nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        device=self.device,
                    )
                ),
                norm_f(
                    nn.Conv2d(
                        32,
                        32,
                        (3, 9),
                        stride=(1, 2),
                        padding=(1, 4),
                        device=self.device,
                    )
                ),
                norm_f(nn.Conv2d(32, 32, (3, 3), padding=(1, 1), device=self.device)),
            ]
        )
        self.conv_post = norm_f(
            nn.Conv2d(32, 1, (3, 3), padding=(1, 1), device=self.device)
        )

    def forward(self, x: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        r"""
        Forward pass of the DiscriminatorR class.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the intermediate feature maps and the output tensor.
        """
        fmap = []

        # Compute the magnitude spectrogram of the input waveform
        x = self.spectrogram(x)

        # Add a channel dimension to the spectrogram tensor
        x = x.unsqueeze(1)

        # Apply the convolutional layers with leaky ReLU activation
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, self.LRELU_SLOPE)
            fmap.append(x)

        # Apply the post-convolutional layer
        x = self.conv_post(x)
        fmap.append(x)

        # Flatten the output tensor
        x = torch.flatten(x, 1, -1)

        return fmap, x

    def spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the magnitude spectrogram of the input waveform.

        Args:
            x (torch.Tensor): Input waveform tensor of shape [B, C, T].

        Returns:
            torch.Tensor: Magnitude spectrogram tensor of shape [B, F, TT], where F is the number of frequency bins and TT is the number of time frames.
        """
        n_fft, hop_length, win_length = self.resolution

        # Apply reflection padding to the input waveform
        x = F.pad(
            x,
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )

        # Squeeze the input waveform to remove the channel dimension
        x = x.squeeze(1)

        # Compute the short-time Fourier transform of the input waveform
        x = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            center=False,
            return_complex=True,
        )  # [B, F, TT, 2]

        # Compute the magnitude spectrogram from the complex spectrogram
        mag = torch.norm(x, p=2, dim=-1)  # [B, F, TT]

        return mag
