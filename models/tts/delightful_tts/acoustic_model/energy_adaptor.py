from typing import Tuple

import torch
from torch import nn

from .helpers import average_over_durations
from .variance_predictor import VariancePredictor


class EnergyAdaptor(nn.Module):
    """Variance Adaptor with an added 1D conv layer. Used to
    get energy embeddings.

    Args:
        channels_in (int): Number of in channels for conv layers.
        channels_out (int): Number of out channels.
        kernel_size (int): Size the kernel for the conv layers.
        dropout (float): Probability of dropout.
        leaky_relu_slope (float): Slope for the leaky relu.
        emb_kernel_size (int): Size the kernel for the pitch embedding.

    Inputs: inputs, mask
        - **inputs** (batch, time1, dim): Tensor containing input vector
        - **target** (batch, 1, time2): Tensor containing the energy target
        - **dr** (batch, time1): Tensor containing aligner durations vector
        - **mask** (batch, time1): Tensor containing indices to be masked
    Returns:
        - **energy prediction** (batch, 1, time1): Tensor produced by energy predictor
        - **energy embedding** (batch, channels, time1): Tensor produced energy adaptor
        - **average energy target(train only)** (batch, 1, time1): Tensor produced after averaging over durations

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
        self.energy_predictor = VariancePredictor(
            channels_in=channels_in,
            channels=channels_hidden,
            channels_out=channels_out,
            kernel_size=kernel_size,
            p_dropout=dropout,
            leaky_relu_slope=leaky_relu_slope,
        )
        self.energy_emb = nn.Conv1d(
            1,
            channels_hidden,
            kernel_size=emb_kernel_size,
            padding=int((emb_kernel_size - 1) / 2),
        )

    def get_energy_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, dr: torch.IntTensor, mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Function is used during training to get the energy prediction, average energy target, and energy embedding.

        Args:
            x (torch.Tensor): A 3D tensor of shape [B, T_src, C] where B is the batch size,
                            T_src is the source sequence length, and C is the number of channels.
            target (torch.Tensor): A 3D tensor of shape [B, 1, T_max2] where B is the batch size,
                                T_max2 is the maximum target sequence length.
            dr (torch.IntTensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the durations.
            mask (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the mask.

        Returns:
            energy_pred (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                        T_src is the source sequence length. The values represent the energy prediction.
            avg_energy_target (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                            T_src is the source sequence length. The values represent the average energy target.
            energy_emb (torch.Tensor): A 3D tensor of shape [B, C, T_src] where B is the batch size,
                                    C is the number of channels, T_src is the source sequence length. The values represent the energy embedding.
        Shapes:
            x: :math: `[B, T_src, C]`
            target: :math: `[B, 1, T_max2]`
            dr: :math: `[B, T_src]`
            mask: :math: `[B, T_src]`
        """
        energy_pred = self.energy_predictor.forward(x, mask)
        energy_pred = energy_pred.unsqueeze(1)

        avg_energy_target = average_over_durations(target, dr)
        energy_emb = self.energy_emb(avg_energy_target)

        return energy_pred, avg_energy_target, energy_emb

    def add_energy_embedding_train(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        dr: torch.IntTensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Add energy embedding during training.

        This method calculates the energy embedding and adds it to the input tensor 'x'.
        It also returns the predicted energy and the average target energy.

        Args:
            x (torch.Tensor): The input tensor to which the energy embedding will be added.
            target (torch.Tensor): The target tensor used in the energy embedding calculation.
            dr (torch.IntTensor): The duration tensor used in the energy embedding calculation.
            mask (torch.Tensor): The mask tensor used in the energy embedding calculation.

        Returns:
            x (torch.Tensor): The input tensor with added energy embedding.
            energy_pred (torch.Tensor): The predicted energy tensor.
            avg_energy_target (torch.Tensor): The average target energy tensor.
        """
        energy_pred, avg_energy_target, energy_emb = self.get_energy_embedding_train(
            x=x,
            target=target,
            dr=dr,
            mask=mask,
        )
        x = x + energy_emb.transpose(1, 2)
        return x, energy_pred, avg_energy_target

    def get_energy_embedding(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Function is used during inference to get the energy embedding and energy prediction.

        Args:
            x (torch.Tensor): A 3D tensor of shape [B, T_src, C] where B is the batch size,
                            T_src is the source sequence length, and C is the number of channels.
            mask (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the mask.

        Returns:
            energy_emb_pred (torch.Tensor): A 3D tensor of shape [B, C, T_src] where B is the batch size,
                                            C is the number of channels, T_src is the source sequence length. The values represent the energy embedding.
            energy_pred (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                        T_src is the source sequence length. The values represent the energy prediction.
        """
        energy_pred = self.energy_predictor.forward(x, mask)
        energy_pred = energy_pred.unsqueeze(1)

        energy_emb_pred = self.energy_emb(energy_pred)
        return energy_emb_pred, energy_pred

    def add_energy_embedding(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Add energy embedding during inference.

        This method calculates the energy embedding and adds it to the input tensor 'x'.
        It also returns the predicted energy.

        Args:
            x (torch.Tensor): The input tensor to which the energy embedding will be added.
            mask (torch.Tensor): The mask tensor used in the energy embedding calculation.
            energy_transform (Callable): A function to transform the energy prediction.

        Returns:
            x (torch.Tensor): The input tensor with added energy embedding.
            energy_pred (torch.Tensor): The predicted energy tensor.
        """
        energy_emb_pred, energy_pred = self.get_energy_embedding(x, mask)
        x = x + energy_emb_pred.transpose(1, 2)
        return x, energy_pred
