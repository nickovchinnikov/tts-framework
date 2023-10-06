from lightning.pytorch import LightningModule
import torch

from .stft_loss import STFTLoss


class MultiResolutionSTFTLoss(LightningModule):
    r"""
    Multi resolution STFT loss module.

    The Multi resolution STFT loss module is a PyTorch module that computes the spectral convergence and log STFT magnitude losses for a predicted signal and a groundtruth signal at multiple resolutions. The module is designed for speech and audio signal processing tasks, such as speech enhancement and source separation.

    The module takes as input a list of tuples, where each tuple contains the FFT size, hop size, and window length for a particular resolution. For each resolution, the module computes the spectral convergence and log STFT magnitude losses using the STFTLoss module, which is a PyTorch module that computes the STFT of a signal and the corresponding magnitude spectrogram.

    The spectral convergence loss measures the similarity between two magnitude spectrograms, while the log STFT magnitude loss measures the similarity between two logarithmically-scaled magnitude spectrograms. The logarithm is applied to the magnitude spectrograms to convert them to a decibel scale, which is more perceptually meaningful than the linear scale.

    The Multi resolution STFT loss module returns the average spectral convergence and log STFT magnitude losses across all resolutions. This allows the module to capture both fine-grained and coarse-grained spectral information in the predicted and groundtruth signals.
    """

    def __init__(
        self,
        resolutions: list[tuple[int, int, int]],
    ):
        r"""
        Initialize Multi resolution STFT loss module.

        Args:
            resolutions (list): List of (FFT size, shift size, window length).
        """
        super().__init__()

        self.stft_losses = torch.nn.ModuleList(
            [STFTLoss(fs, ss, wl) for fs, ss, wl in resolutions]
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = torch.tensor(0.0, device=self.device)
        mag_loss = torch.tensor(0.0, device=self.device)

        # Compute the spectral convergence and log STFT magnitude losses for each resolution
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l

        # Average the losses across all resolutions
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss
