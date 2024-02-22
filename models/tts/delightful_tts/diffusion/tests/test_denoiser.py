import unittest

import torch

from models.tts.delightful_tts.diffusion.denoiser import Denoiser


class TestDenoiser(unittest.TestCase):
    def test_forward(self):
        preprocess_config = {
            "preprocessing": {
                "mel": {"n_mel_channels": 8},
            },
        }
        model_config = {
            "transformer": {"encoder_hidden": 16, "speaker_embed_dim": 8},
            "denoiser": {"residual_channels": 8, "residual_layers": 4, "denoiser_dropout": 0.1},
            "multi_speaker": True,
        }
        denoiser = Denoiser(preprocess_config, model_config)

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
