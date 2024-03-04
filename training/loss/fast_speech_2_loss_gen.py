from typing import Tuple

from piq import SSIMLoss
import torch
from torch import nn
from torch.nn import Module

from training.loss.bin_loss import BinLoss
from training.loss.forward_sum_loss import ForwardSumLoss
from training.loss.utils import sample_wise_min_max

from .spectral_convergence_loss import SpectralConvergengeLoss
from .stft_magnitude_loss import STFTMagnitudeLoss


class FastSpeech2LossGen(Module):
    def __init__(
        self,
        bin_warmup: bool = False,
    ):
        r"""Initializes the FastSpeech2LossGen module.

        Args:
            bin_warmup (bool, optional): Whether to use binarization warmup. Defaults to False. NOTE: Switch this off if you preload the model with a checkpoint that has already passed the warmup phase.
        """
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.sum_loss = ForwardSumLoss()
        self.bin_loss = BinLoss()

        self.spectral_conv_loss = SpectralConvergengeLoss()

        self.logstft_loss = STFTMagnitudeLoss(
            log=True,
            reduction="mean",
            distance="L1",
        )

        self.bin_warmup = bin_warmup

    def forward(
        self,
        src_masks: torch.Tensor,
        mel_masks: torch.Tensor,
        mel_targets: torch.Tensor,
        mel_predictions: torch.Tensor,
        postnet_outputs: torch.Tensor,
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
        energy_pred: torch.Tensor,
        energy_target: torch.Tensor,
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
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        r"""Computes the loss for the FastSpeech2 model.

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
            energy_pred (torch.Tensor): Predicted energy.
            energy_target (torch.Tensor): Ground-truth energy.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The total loss and its components.

        Note:
            Here is the description of the returned loss components:
            `total_loss`: This is the total loss computed as the sum of all the other losses.
            `mel_loss`: This is the mean absolute error (MAE) loss between the predicted and target mel-spectrograms. It measures how well the model predicts the mel-spectrograms.
            `sc_mag_loss`: This is the spectral convergence loss between the predicted and target mel-spectrograms. It measures how well the model predicts the mel-spectrograms in terms of their spectral structure.
            `log_mag_loss`: This is the log STFT magnitude loss between the predicted and target mel-spectrograms. It measures how well the model predicts the mel-spectrograms in terms of their spectral structure.
            `ssim_loss`: This is the Structural Similarity Index (SSIM) loss between the predicted and target mel-spectrograms. It measures the similarity between the two mel-spectrograms in terms of their structure, contrast, and luminance.
            `duration_loss`: This is the mean squared error (MSE) loss between the predicted and target log-durations. It measures how well the model predicts the durations of the phonemes.
            `u_prosody_loss`: This is the MAE loss between the predicted and reference unvoiced prosody. It measures how well the model predicts the prosody (rhythm, stress, and intonation) of the unvoiced parts of the speech.
            `p_prosody_loss`: This is the MAE loss between the predicted and reference voiced prosody. It measures how well the model predicts the prosody of the voiced parts of the speech.
            `pitch_loss`: This is the MSE loss between the predicted and target pitch. It measures how well the model predicts the pitch of the speech.
            `ctc_loss`: This is the Connectionist Temporal Classification (CTC) loss computed from the log-probability of attention and the lengths of the source sequences and mel-spectrograms. It measures how well the model aligns the input and output sequences.
            `bin_loss`: This is the binarization loss computed from the hard and soft attention. It measures how well the model learns to attend to the correct parts of the input sequence.
            `energy_loss`: This is the MSE loss between the predicted and target energy. It measures how well the model predicts the energy of the speech.
        """
        log_duration_targets = torch.log(durations.float() + 1).to(src_masks.device)

        log_duration_targets.requires_grad = False
        mel_targets.requires_grad = False
        p_targets.requires_grad = False
        energy_target.requires_grad = False

        log_duration_predictions = log_duration_predictions.masked_select(~src_masks)
        log_duration_targets = log_duration_targets.masked_select(~src_masks)

        mel_masks_expanded = mel_masks.unsqueeze(1)

        mel_predictions_normalized = sample_wise_min_max(mel_predictions).float().to(mel_predictions.device)
        mel_targets_normalized = sample_wise_min_max(mel_targets).float().to(mel_predictions.device)
        postnet_outputs_normalized = sample_wise_min_max(postnet_outputs).float().to(mel_predictions.device)

        ssim_loss: torch.Tensor = self.ssim_loss(
            mel_predictions_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1),
        )

        ssim_loss_postnet: torch.Tensor = self.ssim_loss(
            postnet_outputs_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1),
        )

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            # print(
            #     f"Overflow in ssim loss detected, which was {ssim_loss.item()}, setting to 1.0",
            # )
            ssim_loss = torch.tensor([1.0], device=mel_predictions.device)

        masked_mel_predictions = mel_predictions.masked_select(~mel_masks_expanded)

        masked_mel_targets = mel_targets.masked_select(~mel_masks_expanded)

        masked_postnet_outputs = postnet_outputs.masked_select(~mel_masks_expanded)

        mel_loss: torch.Tensor = self.mae_loss(masked_mel_predictions, masked_mel_targets)
        mel_loss_postnet: torch.Tensor = self.mae_loss(masked_postnet_outputs, masked_mel_targets)

        # sc_mag_loss = self.spectral_conv_loss(mel_predictions, mel_targets)
        # log_mag_loss = self.logstft_loss(mel_predictions, mel_targets)

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
            u_prosody_ref, u_prosody_pred,
        )

        duration_loss: torch.Tensor = self.mse_loss(
            log_duration_predictions, log_duration_targets,
        )

        pitch_predictions = pitch_predictions.masked_select(~src_masks)
        p_targets = p_targets.masked_select(~src_masks)

        pitch_loss: torch.Tensor = self.mse_loss(pitch_predictions, p_targets)

        ctc_loss: torch.Tensor = self.sum_loss(
            attn_logprob=attn_logprob, in_lens=src_lens, out_lens=mel_lens,
        )

        if self.bin_warmup:
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
        else:
            bin_loss: torch.Tensor = self.bin_loss(hard_attention=attn_hard, soft_attention=attn_soft)

        energy_loss: torch.Tensor = self.mse_loss(energy_pred, energy_target)

        total_loss = (
            mel_loss
            + mel_loss_postnet
            + duration_loss
            + u_prosody_loss
            + p_prosody_loss
            + ssim_loss
            + ssim_loss_postnet
            + pitch_loss
            + ctc_loss
            + bin_loss
            + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            mel_loss_postnet,
            ssim_loss,
            ssim_loss_postnet,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
            energy_loss,
        )
