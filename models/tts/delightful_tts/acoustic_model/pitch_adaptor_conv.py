from typing import Tuple

import torch
from torch import nn

from .helpers import average_over_durations
from .variance_predictor import VariancePredictor


class PitchAdaptorConv(nn.Module):
    """The PitchAdaptorConv class is a pitch adaptor network in the model.
    Updated version of the PitchAdaptorConv uses the conv embeddings for the pitch.

    Args:
        channels_in (int): Number of in channels for conv layers.
        channels_out (int): Number of out channels.
        kernel_size (int): Size the kernel for the conv layers.
        dropout (float): Probability of dropout.
        leaky_relu_slope (float): Slope for the leaky relu.
        emb_kernel_size (int): Size the kernel for the pitch embedding.

    Inputs: inputs, mask
        - **inputs** (batch, time1, dim): Tensor containing input vector
        - **target** (batch, 1, time2): Tensor containing the pitch target
        - **dr** (batch, time1): Tensor containing aligner durations vector
        - **mask** (batch, time1): Tensor containing indices to be masked
    Returns:
        - **pitch prediction** (batch, 1, time1): Tensor produced by pitch predictor
        - **pitch embedding** (batch, channels, time1): Tensor produced pitch adaptor
        - **average pitch target(train only)** (batch, 1, time1): Tensor produced after averaging over durations

    """

    def __init__(
        self,
        channels_in: int,
        channels_hidden: int,
        channels_out: int,
        kernel_size: int,
        dropout: float,
        leaky_relu_slope: float,
        emb_kernel_size: int,
    ):
        super().__init__()
        self.pitch_predictor = VariancePredictor(
            channels_in=channels_in,
            channels=channels_hidden,
            channels_out=channels_out,
            kernel_size=kernel_size,
            p_dropout=dropout,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.pitch_emb = nn.Conv1d(
            1,
            channels_hidden,
            kernel_size=emb_kernel_size,
            padding=int((emb_kernel_size - 1) / 2),
        )

    def get_pitch_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, dr: torch.Tensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Function is used during training to get the pitch prediction, average pitch target,
        and pitch embedding.

        Args:
            x (torch.Tensor): A 3D tensor of shape [B, T_src, C] where B is the batch size,
                            T_src is the source sequence length, and C is the number of channels.
            target (torch.Tensor): A 3D tensor of shape [B, 1, T_max2] where B is the batch size,
                                T_max2 is the maximum target sequence length.
            dr (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the durations.
            mask (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the mask.

        Returns:
            pitch_pred (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                        T_src is the source sequence length. The values represent the pitch prediction.
            avg_pitch_target (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                            T_src is the source sequence length. The values represent the average pitch target.
            pitch_emb (torch.Tensor): A 3D tensor of shape [B, C, T_src] where B is the batch size,
                                    C is the number of channels, T_src is the source sequence length. The values represent the pitch embedding.
        Shapes:
            x: :math: `[B, T_src, C]`
            target: :math: `[B, 1, T_max2]`
            dr: :math: `[B, T_src]`
            mask: :math: `[B, T_src]`
        """
        pitch_pred = self.pitch_predictor.forward(x, mask)
        pitch_pred = pitch_pred.unsqueeze(1)

        avg_pitch_target = average_over_durations(target, dr)
        pitch_emb = self.pitch_emb(avg_pitch_target)

        return pitch_pred, avg_pitch_target, pitch_emb

    def add_pitch_embedding_train(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        dr: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Add pitch embedding during training.

        This method calculates the pitch embedding and adds it to the input tensor 'x'.
        It also returns the predicted pitch and the average target pitch.

        Args:
            x (torch.Tensor): The input tensor to which the pitch embedding will be added.
            target (torch.Tensor): The target tensor used in the pitch embedding calculation.
            dr (torch.Tensor): The duration tensor used in the pitch embedding calculation.
            mask (torch.Tensor): The mask tensor used in the pitch embedding calculation.

        Returns:
            x (torch.Tensor): The input tensor with added pitch embedding.
            pitch_pred (torch.Tensor): The predicted pitch tensor.
            avg_pitch_target (torch.Tensor): The average target pitch tensor.
        """
        pitch_pred, avg_pitch_target, pitch_emb = self.get_pitch_embedding_train(
            x=x,
            target=target.unsqueeze(1),
            dr=dr,
            mask=mask,
        )
        x = x + pitch_emb.transpose(1, 2)
        return x, pitch_pred, avg_pitch_target

    def get_pitch_embedding(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Function is used during inference to get the pitch embedding and pitch prediction.

        Args:
            x (torch.Tensor): A 3D tensor of shape [B, T_src, C] where B is the batch size,
                            T_src is the source sequence length, and C is the number of channels.
            mask (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the mask.

        Returns:
            pitch_emb_pred (torch.Tensor): A 3D tensor of shape [B, C, T_src] where B is the batch size,
                                            C is the number of channels, T_src is the source sequence length. The values represent the pitch embedding.
            pitch_pred (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                        T_src is the source sequence length. The values represent the pitch prediction.
        """
        pitch_pred = self.pitch_predictor.forward(x, mask)
        pitch_pred = pitch_pred.unsqueeze(1)

        pitch_emb_pred = self.pitch_emb(pitch_pred)
        return pitch_emb_pred, pitch_pred

    def add_pitch_embedding(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Add pitch embedding during inference.

        This method calculates the pitch embedding and adds it to the input tensor 'x'.
        It also returns the predicted pitch.

        Args:
            x (torch.Tensor): The input tensor to which the pitch embedding will be added.
            mask (torch.Tensor): The mask tensor used in the pitch embedding calculation.
            pitch_transform (Callable): A function to transform the pitch prediction.

        Returns:
            x (torch.Tensor): The input tensor with added pitch embedding.
            pitch_pred (torch.Tensor): The predicted pitch tensor.
        """
        pitch_emb_pred, pitch_pred = self.get_pitch_embedding(x, mask)
        x = x + pitch_emb_pred.transpose(1, 2)
        return x, pitch_pred
