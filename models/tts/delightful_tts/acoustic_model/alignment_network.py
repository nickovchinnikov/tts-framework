from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


def maximum_path(
    value: Tensor,
    mask: Tensor,
    max_neg_val: Optional[float] = None,
):
    """Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[
            :,
            :-1,
        ]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1  # type: ignore
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)  # type: ignore
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


class AlignmentNetwork(torch.nn.Module):
    r"""Aligner Network for learning alignment between the input text and the model output with Gaussian Attention.

    The architecture is as follows:
    query -> conv1d -> relu -> conv1d -> relu -> conv1d -> L2_dist -> softmax -> alignment
    key   -> conv1d -> relu -> conv1d -----------------------^

    Args:
        in_query_channels (int): Number of channels in the query network.
        in_key_channels (int): Number of channels in the key network.
        attn_channels (int): Number of inner channels in the attention layers.
        temperature (float, optional): Temperature for the softmax. Defaults to 0.0005.
    """

    def __init__(
        self,
        in_query_channels: int,
        in_key_channels: int,
        attn_channels: int,
        temperature: float = 0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_layer = nn.Sequential(
            nn.Conv1d(
                in_key_channels,
                in_key_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            nn.Conv1d(
                in_key_channels * 2,
                attn_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )

        self.query_layer = nn.Sequential(
            nn.Conv1d(
                in_query_channels,
                in_query_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            nn.Conv1d(
                in_query_channels * 2,
                in_query_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
            torch.nn.ReLU(),
            nn.Conv1d(
                in_query_channels,
                attn_channels,
                kernel_size=1,
                padding=0,
                bias=True,
            ),
        )

        self.init_layers()

    def init_layers(self):
        r"""Initialize the weights of the key and query layers using Xavier uniform initialization.

        The gain is calculated based on the activation function: ReLU for the first layer and linear for the rest.
        """
        torch.nn.init.xavier_uniform_(
            self.key_layer[0].weight,
            gain=torch.nn.init.calculate_gain("relu"),
        )
        torch.nn.init.xavier_uniform_(
            self.key_layer[2].weight,
            gain=torch.nn.init.calculate_gain("linear"),
        )
        torch.nn.init.xavier_uniform_(
            self.query_layer[0].weight,
            gain=torch.nn.init.calculate_gain("relu"),
        )
        torch.nn.init.xavier_uniform_(
            self.query_layer[2].weight,
            gain=torch.nn.init.calculate_gain("linear"),
        )
        torch.nn.init.xavier_uniform_(
            self.query_layer[4].weight,
            gain=torch.nn.init.calculate_gain("linear"),
        )

    def _forward_aligner(
        self,
        queries: Tensor,
        keys: Tensor,
        mask: Optional[Tensor] = None,
        attn_prior: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""Forward pass of the aligner encoder.

        Args:
            queries (Tensor): Input queries of shape [B, C, T_de].
            keys (Tensor): Input keys of shape [B, C_emb, T_en].
            mask (Optional[Tensor], optional): Mask of shape [B, T_de]. Defaults to None.
            attn_prior (Optional[Tensor], optional): Prior attention tensor. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the soft attention mask of shape [B, 1, T_en, T_de] and
            log probabilities of shape [B, 1, T_en , T_de].
        """
        key_out = self.key_layer(keys)
        query_out = self.query_layer(queries)
        attn_factor = (query_out[:, :, :, None] - key_out[:, :, None]) ** 2
        attn_logp = -self.temperature * attn_factor.sum(1, keepdim=True)
        if attn_prior is not None:
            attn_logp = self.log_softmax(attn_logp) + torch.log(
                attn_prior[:, None] + 1e-8,
            ).permute((0, 1, 3, 2))

        if mask is not None:
            attn_logp.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))

        attn = self.softmax(attn_logp)
        return attn, attn_logp

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        x_mask: Tensor,
        y_mask: Tensor,
        attn_priors: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        r"""Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.Tensor): Input sequence.
            y (torch.Tensor): Output sequence.
            x_mask (torch.Tensor): Input sequence mask.
            y_mask (torch.Tensor): Output sequence mask.
            attn_priors (torch.Tensor): Prior for the aligner network map.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`
            - attn_priors: :math:`[B, T_de, T_en]`

            - aligner_durations: :math:`[B, T_en]`
            - aligner_soft: :math:`[B, T_de, T_en]`
            - aligner_logprob: :math:`[B, 1, T_de, T_en]`
            - aligner_mas: :math:`[B, T_de, T_en]`
        """
        # [B, 1, T_en, T_de]
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)

        aligner_soft, aligner_logprob = self._forward_aligner(
            y.transpose(1, 2),
            x.transpose(1, 2),
            x_mask,
            attn_priors,
        )

        aligner_mas = maximum_path(
            aligner_soft.squeeze(1).transpose(1, 2).contiguous(),
            attn_mask.squeeze(1).contiguous(),
        )
        aligner_durations = torch.sum(aligner_mas, -1).int()

        # [B, T_max2, T_max]
        aligner_soft = aligner_soft.squeeze(1)
        # [B, T_max, T_max2] -> [B, T_max2, T_max]
        aligner_mas = aligner_mas.transpose(1, 2)

        return aligner_logprob, aligner_soft, aligner_mas, aligner_durations
