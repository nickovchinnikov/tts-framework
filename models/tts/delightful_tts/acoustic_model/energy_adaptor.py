from typing import Callable, Tuple

import torch
from torch import nn

from .variance_predictor import VariancePredictor


def average_over_durations(values: torch.Tensor, durs: torch.Tensor) -> torch.Tensor:
    r"""Function calculates the average of values over specified durations.

    Args:
    values (torch.Tensor): A 3D tensor of shape [B, 1, T_de] where B is the batch size,
                           T_de is the duration of each element in the batch. The values
                           represent some quantity that needs to be averaged over durations.
    durs (torch.Tensor): A 2D tensor of shape [B, T_en] where B is the batch size,
                         T_en is the number of elements in each batch. The values represent
                         the durations over which the averaging needs to be done.

    Returns:
    avg (torch.Tensor): A 3D tensor of shape [B, 1, T_en] where B is the batch size,
                        T_en is the number of elements in each batch. The values represent
                        the average of the input values over the specified durations.

    Note:
    The function uses PyTorch operations for efficient computation on GPU.

    Shapes:
        - values: :math:`[B, 1, T_de]`
        - durs: :math:`[B, T_en]`
        - avg: :math:`[B, 1, T_en]`
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).float()
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).float()

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg


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
        energy_pred.unsqueeze(1)

        avg_energy_target = average_over_durations(target, dr)
        energy_emb = self.energy_emb(avg_energy_target)

        return energy_pred, avg_energy_target, energy_emb

    def get_energy_embedding(self, x: torch.Tensor, mask: torch.Tensor, energy_transform: Callable) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Function is used during inference to get the energy embedding and energy prediction.

        Args:
            x (torch.Tensor): A 3D tensor of shape [B, T_src, C] where B is the batch size,
                            T_src is the source sequence length, and C is the number of channels.
            mask (torch.Tensor): A 2D tensor of shape [B, T_src] where B is the batch size,
                                T_src is the source sequence length. The values represent the mask.
            energy_transform (Callable): A function to transform the energy prediction.

        Returns:
            energy_emb_pred (torch.Tensor): A 3D tensor of shape [B, C, T_src] where B is the batch size,
                                            C is the number of channels, T_src is the source sequence length. The values represent the energy embedding.
            energy_pred (torch.Tensor): A 3D tensor of shape [B, 1, T_src] where B is the batch size,
                                        T_src is the source sequence length. The values represent the energy prediction.
        """
        energy_pred = self.energy_predictor.forward(x, mask)
        energy_pred.unsqueeze(1)

        if energy_transform is not None:
            energy_pred = energy_transform(energy_pred, (~mask).sum(dim=(1, 2)), self.pitch_mean, self.pitch_std)

        energy_emb_pred = self.energy_emb(energy_pred)
        return energy_emb_pred, energy_pred
