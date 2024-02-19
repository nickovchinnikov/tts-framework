import torch
from torch.nn import Module


class SpectralConvergengeLoss(Module):
    r"""Spectral convergence loss module.
    Spectral convergence loss is a measure of the similarity between two magnitude spectrograms.

    The spectral convergence loss is calculated as the Frobenius norm of the difference between the predicted and groundtruth magnitude spectrograms, divided by the Frobenius norm of the groundtruth magnitude spectrogram. The Frobenius norm is a matrix norm that is equivalent to the square root of the sum of the squared elements of a matrix.

    The spectral convergence loss is a useful metric for evaluating the quality of a predicted signal, as it measures the degree to which the predicted signal matches the groundtruth signal in terms of its spectral content. A lower spectral convergence loss indicates a better match between the predicted and groundtruth signals.
    """

    def __init__(self):
        r"""Initilize spectral convergence loss module."""
        super().__init__()

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        r"""Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Spectral convergence loss value.
        """
        # Ensure that x_mag and y_mag have the same size along dimension 1
        min_len = min(x_mag.shape[1], y_mag.shape[1])
        x_mag = x_mag[:, :min_len]
        y_mag = y_mag[:, :min_len]

        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
