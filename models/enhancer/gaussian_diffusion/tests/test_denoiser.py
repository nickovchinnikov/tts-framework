import unittest

import torch

from models.config.configs import DiffusionConfig
from models.enhancer.gaussian_diffusion.denoiser import Denoiser


class TestDenoiser(unittest.TestCase):
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
            # trainsformer params
            encoder_hidden=16,
            decoder_hidden=16,
            speaker_embed_dim=8,
            # loss params
            noise_loss="l1",
        )

        denoiser = Denoiser(model_config)

        B = 2  # Batch size
        M = 8  # Mel channels
        T = 1  # Time steps


        mel = torch.randn(M, M, 16)
        conditioner = torch.randn(T, T, 16)
        diffusion_step = torch.randn(B)
        speaker_emb = torch.randn(T, T, 8)

        output = denoiser(mel, diffusion_step, conditioner, speaker_emb)

        # Test output shape
        self.assertEqual(output.shape, (B, T, 8, 16))

if __name__ == "__main__":
    unittest.main()
