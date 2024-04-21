from typing import List

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss


class DiscriminatorLoss(_Loss):
    """Discriminator Loss module"""

    def forward(
        self,
        disc_real_outputs: List[Tensor],
        disc_generated_outputs: List[Tensor],
    ):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1 - dr) ** 2)
            g_loss = torch.mean(dg**2)
            loss += r_loss + g_loss
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses


class FeatureMatchingLoss(_Loss):
    """Feature Matching Loss module"""

    def forward(self, fmap_r: List[Tensor], fmap_g: List[Tensor]):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss * 2


class GeneratorLoss(_Loss):
    """Generator Loss module"""

    def forward(self, disc_outputs: List[Tensor]):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
