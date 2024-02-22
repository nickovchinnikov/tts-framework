import unittest

import torch
from torch import nn

from models.helpers import (
    tools,
)
from models.tts.delightful_tts.diffusion.gaussian_diffusion import GaussianDiffusion


class TestGaussianDiffusion(unittest.TestCase):
    def test_forward(self):
        args = {
            "model": "shallow",
        }
        preprocess_config = {
            "preprocessing": {
                "mel": {"n_mel_channels": 8},
            },
        }
        model_config = {
            "transformer": {"encoder_hidden": 8, "decoder_hidden": 8, "speaker_embed_dim": 8},
            "denoiser": {
                "residual_channels": 8,
                "residual_layers": 4,
                "denoiser_dropout": 0.1,
                "timesteps": 4,
                "min_beta": 0.1,
                "max_beta": 0.9,
                "s": 0.5,
                "noise_schedule_naive": "vpsde",
                "shallow_timesteps": 1,
            },
            "multi_speaker": True,
        }
        train_config = {
            "loss": {"noise_loss": "l1"},
        }

        mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        diffusion = GaussianDiffusion(args, preprocess_config, model_config, train_config)

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
        self.assertEqual(start_pred[1].shape, (8, 8, 8))
        self.assertEqual(start_pred[2].shape, (8, 8, 8))
        self.assertEqual(start_pred[3].shape, (8, 8, 8))

    # Add more test cases for other methods

if __name__ == "__main__":
    unittest.main()
