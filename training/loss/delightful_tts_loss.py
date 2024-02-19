from typing import Optional

from piq import SSIMLoss
import torch
from torch import Tensor, nn
from torch.nn import functional

from models.config import AcousticModelConfigType
from training.loss.utils import sample_wise_min_max


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length: Tensor, max_len: Optional[int] = None) -> Tensor:
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    max_len_ = max_len if max_len is not None else sequence_length.max().item()

    seq_range = torch.arange(max_len_, dtype=sequence_length.dtype, device=sequence_length.device)

    # B x T_max
    return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)


class ForwardSumLoss(nn.Module):
    r"""A class used to compute the forward sum loss.

    Attributes:
        log_softmax (torch.nn.LogSoftmax): The log softmax function applied along dimension 3.
        ctc_loss (torch.nn.CTCLoss): The CTC loss function with zero infinity set to True.
        blank_logprob (int): The log probability of a blank, default is -1.

    Methods:
        forward(attn_logprob: Tensor, in_lens: Tensor, out_lens: Tensor)
            Compute the forward sum loss.
    """

    def __init__(self, blank_logprob: int = -1):
        r"""Constructs all the necessary attributes for the ForwardSumLoss object.

        Args:
            blank_logprob (int, optional): The log probability of a blank (default is -1).
        """
        super().__init__()
        self.log_softmax = torch.nn.LogSoftmax(dim=3)
        self.ctc_loss = torch.nn.CTCLoss(zero_infinity=True)
        self.blank_logprob = blank_logprob

    def forward(self, attn_logprob: Tensor, in_lens: Tensor, out_lens: Tensor):
        r"""Compute the forward sum loss.

        Args:
            attn_logprob (Tensor): The attention log probabilities.
            in_lens (Tensor): The input lengths.
            out_lens (Tensor): The output lengths.

        Returns:
            total_loss (float): The total loss computed.
        """
        key_lens = in_lens
        query_lens = out_lens
        attn_logprob_padded = functional.pad(input=attn_logprob, pad=(1, 0), value=self.blank_logprob)

        total_loss = 0.0
        for bid in range(attn_logprob.shape[0]):
            target_seq = torch.arange(1, key_lens[bid].item() + 1).unsqueeze(0)
            curr_logprob = attn_logprob_padded[bid].permute(1, 0, 2)[: query_lens[bid], :, : key_lens[bid] + 1]

            curr_logprob = self.log_softmax(curr_logprob[None])[0]
            loss = self.ctc_loss(
                curr_logprob,
                target_seq,
                input_lengths=query_lens[bid : bid + 1],
                target_lengths=key_lens[bid : bid + 1],
            )
            total_loss = total_loss + loss

        total_loss = total_loss / attn_logprob.shape[0]
        return total_loss


class DelightfulTTSLoss(nn.Module):
    r"""A class used to compute the delightful TTS loss.

    Attributes:
        mse_loss (nn.MSELoss): The mean squared error loss function.
        mae_loss (nn.L1Loss): The mean absolute error loss function.
        forward_sum_loss (ForwardSumLoss): The forward sum loss function.
        mel_loss_alpha (float): The weight for the mel loss.
        aligner_loss_alpha (float): The weight for the aligner loss.
        pitch_loss_alpha (float): The weight for the pitch loss.
        energy_loss_alpha (float): The weight for the energy loss.
        u_prosody_loss_alpha (float): The weight for the u prosody loss.
        p_prosody_loss_alpha (float): The weight for the p prosody loss.
        dur_loss_alpha (float): The weight for the duration loss.
        binary_alignment_loss_alpha (float): The weight for the binary alignment loss.

    Methods:
        _binary_alignment_loss(alignment_hard: Tensor, alignment_soft: Tensor)
            Compute the binary alignment loss.
        forward(
            mel_output: Tensor,
            mel_target: Tensor,
            mel_lens: Tensor,
            dur_output: Tensor,
            dur_target: Tensor,
            pitch_output: Tensor,
            pitch_target: Tensor,
            energy_output: Tensor,
            energy_target: Tensor,
            src_lens: Tensor,
            p_prosody_ref: Tensor,
            p_prosody_pred: Tensor,
            u_prosody_ref: Tensor,
            u_prosody_pred: Tensor,
            aligner_logprob: Tensor,
            aligner_hard: Tensor,
            aligner_soft: Tensor,
            binary_loss_weight: Optional[Tensor] = None,
        )
            Compute the delightful TTS loss.
    """

    def __init__(self, config: AcousticModelConfigType):
        r"""Constructs all the necessary attributes for the DelightfulTTSLoss object.

        Args:
            config (AcousticModelConfigType): Configuration parameters for the loss function.
        """
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.forward_sum_loss = ForwardSumLoss()
        self.ssim_loss = SSIMLoss()

        self.mel_loss_alpha = config.loss.mel_loss_alpha
        self.ssim_loss_alpha = config.loss.ssim_loss_alpha
        self.aligner_loss_alpha = config.loss.aligner_loss_alpha
        self.pitch_loss_alpha = config.loss.pitch_loss_alpha
        self.energy_loss_alpha = config.loss.energy_loss_alpha
        self.u_prosody_loss_alpha = config.loss.u_prosody_loss_alpha
        self.p_prosody_loss_alpha = config.loss.p_prosody_loss_alpha
        self.dur_loss_alpha = config.loss.dur_loss_alpha
        self.binary_alignment_loss_alpha = config.loss.binary_align_loss_alpha

    @staticmethod
    def _binary_alignment_loss(alignment_hard: Tensor, alignment_soft: Tensor) -> Tensor:
        """Binary loss that forces soft alignments to match the hard alignments as
        explained in `https://arxiv.org/pdf/2108.10447.pdf`.

        Args:
            alignment_hard (Tensor): The hard alignment tensor.
            alignment_soft (Tensor): The soft alignment tensor.

        Returns:
            loss (float): The computed binary alignment loss.
        """
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()

    def forward(
        self,
        mel_output: Tensor,
        mel_target: Tensor,
        mel_lens: Tensor,
        dur_output: Tensor,
        dur_target: Tensor,
        pitch_output: Tensor,
        pitch_target: Tensor,
        energy_output: Tensor,
        energy_target: Tensor,
        src_lens: Tensor,
        p_prosody_ref: Tensor,
        p_prosody_pred: Tensor,
        u_prosody_ref: Tensor,
        u_prosody_pred: Tensor,
        aligner_logprob: Tensor,
        aligner_hard: Tensor,
        aligner_soft: Tensor,
    ):
        r"""Compute the delightful TTS loss.

        Args:
            mel_output (Tensor): The mel output tensor.
            mel_target (Tensor): The mel target tensor.
            mel_lens (Tensor): The mel lengths tensor.
            dur_output (Tensor): The duration output tensor.
            dur_target (Tensor): The duration target tensor.
            pitch_output (Tensor): The pitch output tensor.
            pitch_target (Tensor): The pitch target tensor.
            energy_output (Tensor): The energy output tensor.
            energy_target (Tensor): The energy target tensor.
            src_lens (Tensor): The source lengths tensor.
            p_prosody_ref (Tensor): The p prosody reference tensor.
            p_prosody_pred (Tensor): The p prosody prediction tensor.
            u_prosody_ref (Tensor): The u prosody reference tensor.
            u_prosody_pred (Tensor): The u prosody prediction tensor.
            aligner_logprob (Tensor): The aligner log probabilities tensor.
            aligner_hard (Tensor): The hard aligner tensor.
            aligner_soft (Tensor): The soft aligner tensor.

        Returns:
            loss_dict (Tupple): A dictionary containing all the loss values.

        Shapes:
        - mel_output: :math:`(B, C_mel, T_mel)`
        - mel_target: :math:`(B, C_mel, T_mel)`
        - mel_lens: :math:`(B)`
        - dur_output: :math:`(B, T_src)`
        - dur_target: :math:`(B, T_src)`
        - pitch_output: :math:`(B, 1, T_src)`
        - pitch_target: :math:`(B, 1, T_src)`
        - energy_output: :math:`(B, 1, T_src)`
        - energy_target: :math:`(B, 1, T_src)`
        - src_lens: :math:`(B)`
        - p_prosody_ref: :math:`(B, T_src, 4)`
        - p_prosody_pred: :math:`(B, T_src, 4)`
        - u_prosody_ref: :math:`(B, 1, 256)
        - u_prosody_pred: :math:`(B, 1, 256)
        - aligner_logprob: :math:`(B, 1, T_mel, T_src)`
        - aligner_hard: :math:`(B, T_mel, T_src)`
        - aligner_soft: :math:`(B, T_mel, T_src)`
        """
        src_mask = sequence_mask(src_lens).to(mel_output.device)  # (B, T_src)
        mel_mask = sequence_mask(mel_lens).to(mel_output.device)  # (B, T_mel)

        dur_target.requires_grad = False
        mel_target.requires_grad = False
        pitch_target.requires_grad = False

        mel_predictions_normalized = sample_wise_min_max(mel_output).float().to(mel_output.device)
        mel_targets_normalized = sample_wise_min_max(mel_target).float().to(mel_target.device)

        masked_mel_predictions = mel_output.masked_select(mel_mask[:, None])
        mel_targets = mel_target.masked_select(mel_mask[:, None])
        mel_loss = self.mae_loss(masked_mel_predictions, mel_targets) * self.mel_loss_alpha

        ssim_loss: torch.Tensor = self.ssim_loss(
            mel_predictions_normalized.unsqueeze(1), mel_targets_normalized.unsqueeze(1),
        ) * self.ssim_loss_alpha

        if ssim_loss.item() > 1.0 or ssim_loss.item() < 0.0:
            print(
                f"Overflow in ssim loss detected, which was {ssim_loss.item()}, setting to 1.0",
            )
            ssim_loss = torch.tensor([1.0], device=mel_output.device)

        p_prosody_ref = p_prosody_ref.detach()
        p_prosody_loss = self.mae_loss(
            p_prosody_ref.masked_select(src_mask.unsqueeze(-1)),
            p_prosody_pred.masked_select(src_mask.unsqueeze(-1)),
        ) * self.p_prosody_loss_alpha

        u_prosody_ref = u_prosody_ref.detach()
        u_prosody_loss = self.mae_loss(u_prosody_ref, u_prosody_pred) * self.u_prosody_loss_alpha

        duration_loss = self.mse_loss(dur_output, dur_target) * self.dur_loss_alpha

        pitch_output = pitch_output.masked_select(src_mask[:, None])
        pitch_target = pitch_target.masked_select(src_mask[:, None])
        pitch_loss = self.mse_loss(pitch_output, pitch_target) * self.pitch_loss_alpha

        energy_output = energy_output.masked_select(src_mask[:, None])
        energy_target = energy_target.masked_select(src_mask[:, None])
        energy_loss = self.mse_loss(energy_output, energy_target) * self.energy_loss_alpha

        forward_sum_loss = self.forward_sum_loss(
            aligner_logprob,
            src_lens,
            mel_lens,
        ) * self.aligner_loss_alpha

        binary_alignment_loss = self._binary_alignment_loss(
            aligner_hard,
            aligner_soft,
        ) * self.binary_alignment_loss_alpha

        total_loss = (
            mel_loss
            + ssim_loss
            + duration_loss
            + u_prosody_loss
            + p_prosody_loss
            + pitch_loss
            + forward_sum_loss
            + binary_alignment_loss
            + energy_loss
        )

        return (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            forward_sum_loss,
            binary_alignment_loss,
            energy_loss,
        )
