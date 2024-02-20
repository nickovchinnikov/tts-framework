from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import Module

from .layers import (
    Block,
    Downsample,
    LinearAttention,
    Mish,
    Residual,
    ResnetBlock,
    Rezero,
    SinusoidalPosEmb,
    Upsample,
)


class GradLogPEstimator2d(Module):
    r"""A PyTorch module for estimating gradients in a 2D space.

    Attributes:
        dim (int): The dimensionality of the input data.
        dim_mults (Tuple[int, ...]): Multipliers for the dimensions.
        groups (int): The number of groups for the GroupNorm.
        n_spks (int): The number of speakers.
        spk_emb_dim (int): The dimensionality of the speaker embeddings.
        n_feats (int): The number of features.
        pe_scale (int): The scale for the positional encoding.
    """

    def __init__(
        self,
        dim: int,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        groups: int = 8,
        n_spks: int = 1,
        spk_emb_dim: int = 64,
        n_feats: int = 80,
        pe_scale: int = 1000,
    ):
        r"""Initializes the GradLogPEstimator2d module.

        Args:
            dim (int): The dimensionality of the input data.
            dim_mults (Tuple[int, ...]): Multipliers for the dimensions.
            groups (int): The number of groups for the GroupNorm.
            n_spks (int): The number of speakers.
            spk_emb_dim (int): The dimensionality of the speaker embeddings.
            n_feats (int): The number of features.
            pe_scale (int): The scale for the positional encoding.
        """
        super().__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks # if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_mlp = nn.Sequential(
                nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                nn.Linear(spk_emb_dim * 4, n_feats),
            )

        self.time_pos_emb = SinusoidalPosEmb(dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim),
        )

        # dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        dims = [2 + (1 if n_spks > 1 else 0), *(dim * m for m in dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                    ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_out))),
                    Downsample(dim_out) if not is_last else nn.Identity(),
                ]),
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                    ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                    Residual(Rezero(LinearAttention(dim_in))),
                    Upsample(dim_in),
                ]),
            )
        self.final_block = Block(dim, dim)
        self.final_conv = nn.Conv2d(dim, 1, 1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        mu: Tensor,
        t: Tensor,
        spk: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Defines the computation performed at every call.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor): The mask tensor.
            mu (Tensor): The mu tensor.
            t (Tensor): The time tensor.
            spk (Optional[Tensor]): The speaker tensor.

        Returns:
            Tensor: The output tensor.
        """
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)
