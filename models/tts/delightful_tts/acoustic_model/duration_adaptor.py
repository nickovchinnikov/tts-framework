from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from models.config import AcousticModelConfigType

from .variance_predictor import VariancePredictor


class DurationAdaptor(Module):
    """DurationAdaptor is a module that adapts the duration of the input sequence.

    Args:
        model_config (AcousticModelConfigType): Configuration object containing model parameters.
    """

    def __init__(
        self,
        model_config: AcousticModelConfigType,
    ):
        super().__init__()
        # Initialize the duration predictor
        self.duration_predictor = VariancePredictor(
            channels_in=model_config.encoder.n_hidden,
            channels=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            p_dropout=model_config.variance_adaptor.p_dropout,
        )

    @staticmethod
    def convert_pad_shape(pad_shape: List[List[int]]) -> List[int]:
        r"""Convert the padding shape from a list of lists to a flat list.

        Args:
            pad_shape (List[List[int]]): Padding shape as a list of lists.

        Returns:
            List[int]: Padding shape as a flat list.
        """
        pad_list = pad_shape[::-1]
        return [item for sublist in pad_list for item in sublist]

    @staticmethod
    def generate_path(duration: Tensor, mask: Tensor) -> Tensor:
        r"""Generate a path based on the duration and mask.

        Args:
            duration (Tensor): Duration tensor.
            mask (Tensor): Mask tensor.

        Returns:
            Tensor: Path tensor.

        Shapes:
        - duration: :math:`[B, T_en]`
        - mask: :math:'[B, T_en, T_de]`
        - path: :math:`[B, T_en, T_de]`
        """
        b, t_x, t_y = mask.shape
        cum_duration = torch.cumsum(duration, 1)

        cum_duration_flat = cum_duration.view(b * t_x)
        path = DurationAdaptor.sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        path = path.view(b, t_x, t_y)
        pad_shape = DurationAdaptor.convert_pad_shape([[0, 0], [1, 0], [0, 0]])
        path = path - F.pad(path, pad_shape)[:, :-1]
        path = path * mask
        return path

    # from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
    @staticmethod
    def sequence_mask(sequence_length: Tensor, max_len: Optional[int] = None) -> Tensor:
        """Create a sequence mask for filtering padding in a sequence tensor.

        Args:
            sequence_length (torch.Tensor): Sequence lengths.
            max_len (int, Optional): Maximum sequence length. Defaults to None.

        Returns:
            torch.Tensor: Sequence mask.

        Shapes:
            - mask: :math:`[B, T_max]`
        """
        if max_len is None:
            max_len = int(sequence_length.max())

        seq_range = torch.arange(
            max_len,
            dtype=sequence_length.dtype,
            device=sequence_length.device,
        )
        # B x T_max
        return seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)

    @staticmethod
    def generate_attn(
        dr: Tensor,
        x_mask: Tensor,
        y_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Generate an attention mask from the linear scale durations.

        Args:
            dr (Tensor): Linear scale durations.
            x_mask (Tensor): Mask for the input (character) sequence.
            y_mask (Tensor): Mask for the output (spectrogram) sequence. Compute it from the predicted durations
                if None. Defaults to None.

        Shapes
           - dr: :math:`(B, T_{en})`
           - x_mask: :math:`(B, T_{en})`
           - y_mask: :math:`(B, T_{de})`
        """
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            sequence_mask = DurationAdaptor.sequence_mask(y_lengths, None)
            y_mask = torch.unsqueeze(sequence_mask, 1).to(dr.dtype)

        # compute the attention mask
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = DurationAdaptor.generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def _expand_encoder_with_durations(
        self,
        encoder_output: Tensor,
        duration_target: Tensor,
        x_mask: Tensor,
        mel_lens: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""Expand the encoder output with durations.

        Args:
            encoder_output (Tensor): Encoder output.
            duration_target (Tensor): Target durations.
            x_mask (Tensor): Mask for the input sequence.
            mel_lens (Tensor): Lengths of the mel spectrograms.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Tuple containing the mask for the output sequence, the attention mask, and the expanded encoder output.
        """
        y_mask = torch.unsqueeze(
            DurationAdaptor.sequence_mask(mel_lens, None),
            1,
        ).to(encoder_output.dtype)
        attn = self.generate_attn(duration_target, x_mask, y_mask)

        encoder_output_ex = torch.einsum(
            "kmn, kmj -> kjn",
            [attn.float(), encoder_output],
        )

        return y_mask, attn, encoder_output_ex

    def forward_train(
        self,
        encoder_output: Tensor,
        encoder_output_res: Tensor,
        duration_target: Tensor,
        src_mask: Tensor,
        mel_lens: Tensor,
    ):
        r"""Forward pass of the DurationAdaptor during training.

        Args:
            encoder_output (Tensor): Encoder output.
            encoder_output_res (Tensor): Encoder output.
            duration_target (Tensor): Target durations.
            src_mask (Tensor): Source mask.
            mel_lens (Tensor): Lengths of the mel spectrograms.

        Returns:
            Tuple: Tuple containing the predicted alignments, log durations, mask for the output sequence, expanded encoder output, and the transposed attention mask.
        """
        log_duration_pred = self.duration_predictor.forward(
            x=encoder_output_res.detach(),
            mask=src_mask,
        )  # [B, C_hidden, T_src] -> [B, T_src]

        y_mask, attn, encoder_output_dr = self._expand_encoder_with_durations(
            encoder_output,
            duration_target,
            x_mask=~src_mask[:, None],
            mel_lens=mel_lens,
        )

        duration_target = torch.log(duration_target + 1)
        duration_pred = torch.exp(log_duration_pred) - 1

        alignments_duration_pred = self.generate_attn(
            duration_pred,
            src_mask.unsqueeze(1),
            y_mask,
        )  # [B, T_max, T_max2']

        return (
            alignments_duration_pred,
            log_duration_pred,
            encoder_output_dr,
            attn.transpose(1, 2),
        )

    def forward(self, encoder_output: Tensor, src_mask: Tensor, d_control: float = 1.0):
        r"""Forward pass of the DurationAdaptor.

        Args:
            encoder_output (Tensor): Encoder output.
            src_mask (Tensor): Source mask.
            d_control (float): Duration control. Defaults to 1.0.

        Returns:
            Tuple: Tuple containing the expanded encoder output, log durations, predicted durations, mask for the output sequence, and the attention mask.
        """
        log_duration_pred = self.duration_predictor(
            x=encoder_output.detach(),
            mask=src_mask,
        )  # [B, C_hidden, T_src] -> [B, T_src]

        duration_pred = (
            (torch.exp(log_duration_pred) - 1) * (~src_mask) * d_control
        )  # -> [B, T_src]

        # duration_pred[duration_pred < 1] = 1.0  # -> [B, T_src]
        duration_pred = torch.where(
            duration_pred < 1,
            torch.ones_like(duration_pred),
            duration_pred,
        )  # -> [B, T_src]

        duration_pred = torch.round(duration_pred)  # -> [B, T_src]
        mel_lens = duration_pred.sum(1)  # -> [B,]

        _, attn, encoder_output_dr = self._expand_encoder_with_durations(
            encoder_output,
            duration_pred.squeeze(1),
            ~src_mask[:, None],
            mel_lens,
        )

        return (
            log_duration_pred,
            encoder_output_dr,
            duration_pred,
            attn.transpose(1, 2),
        )
