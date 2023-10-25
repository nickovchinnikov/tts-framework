from typing import Optional

import librosa
import torch
from torch.nn import Module


class TacotronSTFT(Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        sampling_rate: int,
        center: bool,
        mel_fmax: Optional[int],
        mel_fmin: float = 0.0,
    ):
        r"""TacotronSTFT module that computes mel-spectrograms from a batch of waves.

        Args:
            filter_length (int): Length of the filter window.
            hop_length (int): Number of samples between successive frames.
            win_length (int): Size of the STFT window.
            n_mel_channels (int): Number of mel bins.
            sampling_rate (int): Sampling rate of the input waveforms.
            mel_fmin (int or None): Minimum frequency for the mel filter bank.
            mel_fmax (int or None): Maximum frequency for the mel filter bank.
            center (bool): Whether to pad the input signal on both sides.
        """
        super().__init__()

        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.n_fft = filter_length
        self.hop_size = hop_length
        self.win_size = win_length
        self.fmin = mel_fmin
        self.fmax = mel_fmax
        self.center = center

        # Define the mel filterbank
        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=filter_length,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
        )

        mel_basis = torch.from_numpy(mel).float()

        # Define the Hann window
        hann_window = torch.hann_window(win_length)

        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("hann_window", hann_window)

    def _spectrogram(self, y: torch.Tensor) -> torch.Tensor:
        r"""Computes the linear spectrogram of a batch of waves.

        Args:
            y (torch.Tensor): Input waveforms.

        Returns:
            torch.Tensor: Linear spectrogram.
        """
        assert torch.min(y.data) >= -1
        assert torch.max(y.data) <= 1

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)
        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.hann_window,  # type: ignore
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        return torch.view_as_real(spec)

    def linear_spectrogram(self, y: torch.Tensor) -> torch.Tensor:
        r"""Computes the linear spectrogram of a batch of waves.

        Args:
            y (torch.Tensor): Input waveforms.

        Returns:
            torch.Tensor: Linear spectrogram.
        """
        spec = self._spectrogram(y)
        return torch.norm(spec, p=2, dim=-1)

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Computes mel-spectrograms from a batch of waves.

        Args:
            y (torch.FloatTensor): Input waveforms with shape (B, T) in range [-1, 1]

        Returns:
            torch.FloatTensor: Spectrogram of shape (B, n_spech_channels, T)
            torch.FloatTensor: Mel-spectrogram of shape (B, n_mel_channels, T)
        """
        spec = self._spectrogram(y)

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        mel = torch.matmul(self.mel_basis, spec)  # type: ignore
        mel = self.spectral_normalize_torch(mel)

        return spec, mel

    def spectral_normalize_torch(self, magnitudes: torch.Tensor) -> torch.Tensor:
        r"""Applies dynamic range compression to magnitudes.

        Args:
            magnitudes (torch.Tensor): Input magnitudes.

        Returns:
            torch.Tensor: Output magnitudes.
        """
        return self.dynamic_range_compression_torch(magnitudes)

    def dynamic_range_compression_torch(
        self, x: torch.Tensor, C: int = 1, clip_val: float = 1e-5,
    ) -> torch.Tensor:
        r"""Applies dynamic range compression to x.

        Args:
            x (torch.Tensor): Input tensor.
            C (float): Compression factor.
            clip_val (float): Clipping value.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.log(torch.clamp(x, min=clip_val) * C)

    # NOTE: audio np.ndarray changed to torch.FloatTensor!
    def get_mel_from_wav(self, audio: torch.Tensor) -> torch.Tensor:
        audio_tensor = audio.unsqueeze(0)
        with torch.no_grad():
            _, melspec = self.forward(audio_tensor)
        return melspec.squeeze(0)
