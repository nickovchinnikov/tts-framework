from typing import Dict

import torch
from torch.nn import Module

from training.loss import FastSpeech2LossGen


class AcousticLoss(Module):
    r"""Module that calculates the loss for the Acoustic Model.
    It uses the FastSpeech2LossGen loss function, which is designed for the FastSpeech 2 model.
    This loss function calculates several different types of loss, including reconstruction loss,
    mel loss, SSIM loss, duration loss, utterance prosody loss, phoneme prosody loss, pitch loss,
    CTC loss, and binary classification loss.
    """

    def __init__(self):
        r"""Initializes the AcousticLoss module.
        It initializes the FastSpeech2LossGen loss function and registers buffers for each type of loss.
        """
        super().__init__()

        # Init the loss
        self.loss = FastSpeech2LossGen(fine_tuning=True)

        # Initialize the losses to 0
        self.register_buffer("reconstruction_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("mel_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("ssim_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("duration_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("u_prosody_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("p_prosody_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("pitch_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("ctc_loss", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("bin_loss", torch.tensor([0.0], dtype=torch.float32))

    def forward(
        self,
        src_mask: torch.Tensor,
        src_lens: torch.Tensor,
        mel_mask: torch.Tensor,
        mel_lens: torch.Tensor,
        mels: torch.Tensor,
        y_pred: torch.Tensor,
        log_duration_prediction: torch.Tensor,
        p_prosody_ref: torch.Tensor,
        p_prosody_pred: torch.Tensor,
        pitch_prediction: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        step: int,
    ) -> float:
        r"""Computes the total loss and individual losses for each component of the model.

        Args:
            src_mask (torch.Tensor): The source mask tensor of shape (batch_size, src_len).
            src_lens (torch.Tensor): The source length tensor of shape (batch_size,).
            mel_mask (torch.Tensor): The mel mask tensor of shape (batch_size, mel_len).
            mel_lens (torch.Tensor): The mel length tensor of shape (batch_size,).
            mels (torch.Tensor): The mel target tensor of shape (batch_size, mel_len, mel_dim).
            y_pred (torch.Tensor): The mel prediction tensor of shape (batch_size, mel_len, mel_dim).
            log_duration_prediction (torch.Tensor): The log duration prediction tensor of shape (batch_size, src_len).
            p_prosody_ref (torch.Tensor): The phoneme-level prosody reference tensor of shape (batch_size, src_len).
            p_prosody_pred (torch.Tensor): The phoneme-level prosody prediction tensor of shape (batch_size, src_len).
            pitch_prediction (torch.Tensor): The pitch prediction tensor of shape (batch_size, src_len).
            outputs (Dict[str, torch.Tensor]): A dictionary of model outputs.
            step (int): The current training step.

        Returns:
            float: The total loss.
        """
        (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
        ) = self.loss(
            src_masks=src_mask,
            mel_masks=mel_mask,
            mel_targets=mels,
            mel_predictions=y_pred,
            log_duration_predictions=log_duration_prediction,
            u_prosody_ref=outputs["u_prosody_ref"],
            u_prosody_pred=outputs["u_prosody_pred"],
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_predictions=pitch_prediction,
            p_targets=outputs["pitch_target"],
            durations=outputs["attn_hard_dur"],
            attn_logprob=outputs["attn_logprob"],
            attn_soft=outputs["attn_soft"],
            attn_hard=outputs["attn_hard"],
            src_lens=src_lens,
            mel_lens=mel_lens,
            step=step,
        )

        self.reconstruction_loss += total_loss
        self.mel_loss += mel_loss
        self.ssim_loss += ssim_loss
        self.duration_loss += duration_loss
        self.u_prosody_loss += u_prosody_loss
        self.p_prosody_loss += p_prosody_loss
        self.pitch_loss += pitch_loss
        self.ctc_loss += ctc_loss
        self.bin_loss += bin_loss

        return total_loss
