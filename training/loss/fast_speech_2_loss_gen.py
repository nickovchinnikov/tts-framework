from typing import Tuple

import lightning.pytorch as pl
from piq import SSIMLoss
import torch
import torch.nn as nn

from training.loss.bin_loss import BinLoss
from training.loss.forward_sum_loss import ForwardSumLoss


def sample_wise_min_max(x: torch.Tensor) -> torch.Tensor:
    r"""
    Applies sample-wise min-max normalization to a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_samples, num_features).

    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input tensor.
    """

    # Compute the maximum and minimum values of each sample in the batch
    maximum = torch.amax(x, dim=(1, 2), keepdim=True)
    minimum = torch.amin(x, dim=(1, 2), keepdim=True)

    # Apply sample-wise min-max normalization to the input tensor
    normalized = (x - minimum) / (maximum - minimum)

    return normalized


class FastSpeech2LossGen(pl.LightningModule):
    def __init__(self, fine_tuning: bool):
        r"""
        Initializes the FastSpeech2LossGen module.

        Args:
            fine_tuning (bool): Whether the module is used for fine-tuning.
        """
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()
        self.fine_tuning = fine_tuning

    def forward(
        self,
        src_masks: torch.Tensor,
        mel_masks: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_predictions: torch.Tensor,
        log_duration_predictions: torch.Tensor,
        u_prosody_ref: torch.Tensor,
        u_prosody_pred: torch.Tensor,
        p_prosody_ref: torch.Tensor,
        p_prosody_pred: torch.Tensor,
        durations: torch.Tensor,
        pitch_predictions: torch.Tensor,
        p_targets: torch.Tensor,
        attn_logprob: torch.Tensor,
        attn_soft: torch.Tensor,
        attn_hard: torch.Tensor,
        step: int,
        src_lens: torch.Tensor,
        mel_lens: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r"""
        Computes the loss for the FastSpeech2 model.

        Args:
            src_masks (torch.Tensor): Mask for the source sequence.
            mel_masks (torch.Tensor): Mask for the mel-spectrogram.
            mel_targets (torch.Tensor): Target mel-spectrogram.
            mel_predictions (torch.Tensor): Predicted mel-spectrogram.
            log_duration_predictions (torch.Tensor): Predicted log-duration.
            u_prosody_ref (torch.Tensor): Reference unvoiced prosody.
            u_prosody_pred (torch.Tensor): Predicted unvoiced prosody.
            p_prosody_ref (torch.Tensor): Reference voiced prosody.
            p_prosody_pred (torch.Tensor): Predicted voiced prosody.
            durations (torch.Tensor): Ground-truth durations.
            pitch_predictions (torch.Tensor): Predicted pitch.
            p_targets (torch.Tensor): Ground-truth pitch.
            attn_logprob (torch.Tensor): Log-probability of attention.
            attn_soft (torch.Tensor): Soft attention.
            attn_hard (torch.Tensor): Hard attention.
            step (int): Current training step.
            src_lens (torch.Tensor): Lengths of the source sequences.
            mel_lens (torch.Tensor): Lengths of the mel-spectrograms.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The total loss and its components.
        """
        log_duration_targets = torch.log(durations.float() + 1)

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False
        p_targets.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(~src_masks)
        log_duration_targets = log_duration_targets.masked_select(~src_masks)

        mel_masks_expanded = mel_masks.unsqueeze(1)

        mel_predictions_normalized = sample_wise_min_max(mel_predictions)
        mel_targets_normalized = sample_wise_min_max(mel_targets)

        ssim_loss: torch.Tensor = self.ssim_loss(
            mel_predictions_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1)
        )

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            print(
                f"Overflow in ssim loss detected, which was {ssim_loss.item()}, setting to 1.0"
            )
            ssim_loss = torch.FloatTensor([1.0]).to(self.device)

        masked_mel_predictions = mel_predictions.masked_select(~mel_masks_expanded)

        mel_targets = mel_targets.masked_select(~mel_masks_expanded)

        mel_loss: torch.Tensor = self.mae_loss(masked_mel_predictions, mel_targets)

        p_prosody_ref = p_prosody_ref.permute((0, 2, 1))
        p_prosody_pred = p_prosody_pred.permute((0, 2, 1))

        p_prosody_ref = p_prosody_ref.masked_fill(src_masks.unsqueeze(1), 0.0)
        p_prosody_pred = p_prosody_pred.masked_fill(src_masks.unsqueeze(1), 0.0)

        p_prosody_ref = p_prosody_ref.detach()

        p_prosody_loss: torch.Tensor = 0.5 * self.mae_loss(
            p_prosody_ref.masked_select(~src_masks.unsqueeze(1)),
            p_prosody_pred.masked_select(~src_masks.unsqueeze(1)),
        )

        u_prosody_ref = u_prosody_ref.detach()
        u_prosody_loss: torch.Tensor = 0.5 * self.mae_loss(
            u_prosody_ref, u_prosody_pred
        )

        duration_loss: torch.Tensor = self.mse_loss(
            log_duration_predictions, log_duration_targets
        )

        pitch_predictions = pitch_predictions.masked_select(~src_masks)
        p_targets = p_targets.masked_select(~src_masks)

        pitch_loss: torch.Tensor = self.mse_loss(pitch_predictions, p_targets)

        ctc_loss: torch.Tensor = self.sum_loss(
            attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens
        )

        binarization_loss_enable_steps = 18000
        binarization_loss_warmup_steps = 10000

        if step < binarization_loss_enable_steps:
            bin_loss_weight = 0.0
        else:
            bin_loss_weight = (
                min(
                    (step - binarization_loss_enable_steps)
                    / binarization_loss_warmup_steps,
                    1.0,
                )
                * 1.0
            )
        bin_loss: torch.Tensor = (
            self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)
            * bin_loss_weight
        )

        total_loss = (
            mel_loss
            + duration_loss
            + u_prosody_loss
            + p_prosody_loss
            + ssim_loss
            + pitch_loss
            + ctc_loss
            + bin_loss
        )

        return (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
        )
