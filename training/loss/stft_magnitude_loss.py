import torch


class STFTMagnitudeLoss(torch.nn.Module):
    """STFT magnitude loss module.

    See [Arik et al., 2018](https://arxiv.org/abs/1808.06719)
    and [Engel et al., 2020](https://arxiv.org/abs/2001.04643v1)

    Args:
        log (bool, optional): Log-scale the STFT magnitudes,
            or use linear scale. Default: True
        distance (str, optional): Distance function ["L1", "L2"]. Default: "L1"
        reduction (str, optional): Reduction of the loss elements. Default: "mean"
    """

    def __init__(
        self,
        log: bool = True,
        distance: str = "L1",
        reduction: str = "mean",
        epsilon: float = 1e-8,
    ):
        super().__init__()

        self.log = log
        self.epsilon = epsilon

        if distance == "L1":
            self.distance = torch.nn.L1Loss(reduction=reduction)
        elif distance == "L2":
            self.distance = torch.nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Invalid distance: '{distance}'.")

    def forward(self, x_mag: torch.Tensor, y_mag: torch.Tensor) -> torch.Tensor:
        r"""Calculate forward propagation.

        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).

        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        if self.log:
            x_mag = torch.sign(x_mag) * torch.log(torch.abs(x_mag + self.epsilon))
            y_mag = torch.sign(y_mag) * torch.log(torch.abs(y_mag + self.epsilon))

        return self.distance(x_mag, y_mag)
