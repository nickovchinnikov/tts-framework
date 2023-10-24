import torch
from torch.nn import Module

from .log_stft_magnitude_loss import LogSTFTMagnitudeLoss
from .spectral_convergence_loss import SpectralConvergengeLoss
from .stft import stft


class STFTLoss(Module):
    r"""STFT loss module.

    STFT loss is a combination of two loss functions: the spectral convergence loss and the log STFT magnitude loss.

    The spectral convergence loss measures the similarity between two magnitude spectrograms, while the log STFT magnitude loss measures the similarity between two logarithmically-scaled magnitude spectrograms. The logarithm is applied to the magnitude spectrograms to convert them to a decibel scale, which is more perceptually meaningful than the linear scale.

    The STFT loss is a useful metric for evaluating the quality of a predicted signal, as it measures the degree to which the predicted signal matches the groundtruth signal in terms of its spectral content on both a linear and decibel scale. A lower STFT loss indicates a better match between the predicted and groundtruth signals.

    Args:
        fft_size (int): FFT size.
        shift_size (int): Shift size.
        win_length (int): Window length.
    """

    def __init__(
        self,
        fft_size: int = 1024,
        shift_size: int = 120,
        win_length: int = 600,
    ):
        r"""Initialize STFT loss module."""
        super().__init__()

        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length

        self.register_buffer("window", torch.hann_window(win_length))

        self.spectral_convergenge_loss = SpectralConvergengeLoss()
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss()

    def forward(
        self, x: torch.Tensor, y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.shift_size, self.win_length, self.window)

        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss
