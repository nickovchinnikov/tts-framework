import torch
import torch.nn.functional as F


class LogSTFTMagnitudeLoss(torch.nn.Module):
    r"""
    Log STFT magnitude loss module.
    Log STFT magnitude loss is a loss function that is commonly used in speech and audio signal processing tasks, such as speech enhancement and source separation. It is a modification of the spectral convergence loss, which measures the similarity between two magnitude spectrograms.

    The log STFT magnitude loss is calculated as the mean squared error between the logarithm of the predicted and groundtruth magnitude spectrograms. The logarithm is applied to the magnitude spectrograms to convert them to a decibel scale, which is more perceptually meaningful than the linear scale. The mean squared error is used to penalize large errors between the predicted and groundtruth spectrograms.
    """

    def __init__(self):
        r"""Initilize los STFT magnitude loss module."""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        r"""Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        return F.l1_loss(torch.log(y_mag), torch.log(x_mag))
