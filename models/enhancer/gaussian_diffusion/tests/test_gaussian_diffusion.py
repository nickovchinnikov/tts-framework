import unittest

import torch
from torch import nn

from models.config.configs import DiffusionConfig
from models.enhancer.gaussian_diffusion.gaussian_diffusion import GaussianDiffusion
from models.helpers import (
    tools,
)


class TestGaussianDiffusion(unittest.TestCase):
    def test_forward(self):
        model_config = DiffusionConfig(
            # model parameters
            model="shallow",
            n_mel_channels=8,
            multi_speaker=True,
            # denoiser parameters
            residual_channels=8,
            residual_layers=4,
            denoiser_dropout=0.2,
            noise_schedule_naive="vpsde",
            timesteps=4,
            shallow_timesteps=1,
            min_beta=0.1,
            max_beta=40,
            s=0.008,
            keep_bins=80,
            pe_scale=1000,
            # trainsformer params
            encoder_hidden=8,
            decoder_hidden=8,
            speaker_embed_dim=8,
            # loss params
            noise_loss="l1",
        )

        mel_linear = nn.Linear(
            model_config.decoder_hidden,
            model_config.n_mel_channels,
        )

        diffusion = GaussianDiffusion(model_config)

        mel = torch.randn(1, 8, 8)
        conditioner = torch.randn(1, 8, 8)
        speaker_emb = torch.randn(1, 8, 8)

        mel_mask = tools.get_mask_from_lengths(
            torch.cat([torch.randint(1, 8, (7,)), torch.tensor([8])]),
        )

        coarse_mel = mel_linear(mel)

        start_pred = diffusion.forward(mel, conditioner, speaker_emb, mel_mask, coarse_mel)

        # Test shape
        self.assertEqual(start_pred[0].shape, (8, 8, 8))
        self.assertEqual(start_pred[1].shape, (8, 8, 8)) # type: ignore
        self.assertEqual(start_pred[2].shape, (8, 8, 8)) # type: ignore
        self.assertEqual(start_pred[3].shape, (8, 8, 8)) # type: ignore

if __name__ == "__main__":
    unittest.main()
