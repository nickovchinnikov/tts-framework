import lightning.pytorch as pl
import torch


class BinLoss(pl.LightningModule):
    r"""Binary cross-entropy loss for hard and soft attention.

    Attributes
        None

    Methods
        forward: Computes the binary cross-entropy loss for hard and soft attention.

    """

    def __init__(self):
        super().__init__()

    def forward(
        self, hard_attention: torch.Tensor, soft_attention: torch.Tensor,
    ) -> torch.Tensor:
        r"""Computes the binary cross-entropy loss for hard and soft attention.

        Args:
            hard_attention (torch.Tensor): A binary tensor indicating the hard attention.
            soft_attention (torch.Tensor): A tensor containing the soft attention probabilities.

        Returns:
            torch.Tensor: The binary cross-entropy loss.

        """
        log_sum = torch.log(
            torch.clamp(soft_attention[hard_attention == 1], min=1e-12),
        ).sum()
        return -log_sum / hard_attention.sum()
