import unittest

import torch
from torch.nn.functional import softplus

from models.enhancer.gaussian_diffusion.layers import (
    ConvNorm,
    DiffusionEmbedding,
    LinearNorm,
    Mish,
    ResidualBlock,
)


class TestYourModule(unittest.TestCase):
    def test_mish_activation(self):
        activation = Mish()
        input_tensor = torch.tensor([0.5])
        output_tensor = activation(input_tensor)
        expected_output = input_tensor * torch.tanh(softplus(input_tensor))
        self.assertTrue(torch.allclose(output_tensor, expected_output))

    def test_conv_norm(self):
        # Testing with valid input
        in_channels = 3
        out_channels = 5
        kernel_size = 3
        signal = torch.randn(2, 3, 10)
        conv_norm = ConvNorm(in_channels, out_channels, kernel_size)
        output = conv_norm(signal)
        self.assertEqual(output.shape, (2, 5, 10))

        # Testing with default parameters
        conv_norm = ConvNorm(in_channels, out_channels)
        output = conv_norm(signal)
        self.assertEqual(output.shape, (2, 5, 10))

    def test_diffusion_embedding(self):
        d_denoiser = 8
        embedding = DiffusionEmbedding(d_denoiser)
        input_tensor = torch.randn(2, 4)
        output = embedding(input_tensor)
        self.assertEqual(output.shape, (2, 1, d_denoiser))

    def test_linear_norm(self):
        in_features = 4
        out_features = 6
        linear_norm = LinearNorm(in_features, out_features)
        input_tensor = torch.randn(2, in_features)
        output = linear_norm(input_tensor)
        self.assertEqual(output.shape, (2, out_features))

    def test_residual_block(self):
        d_encoder = 16
        residual_channels = 8
        dropout = 0.1
        d_spk_prj = 8
        multi_speaker = True

        block = ResidualBlock(d_encoder, residual_channels, dropout, d_spk_prj, multi_speaker)

        x = torch.randn(1, 16)
        conditioner = torch.randn(1, 1, 16)
        diffusion_step = torch.randn(1, 8)
        speaker_emb = torch.randn(1, 1, 8)

        output, skip = block(x, conditioner, diffusion_step, speaker_emb)

        self.assertEqual(output.shape, (1, 8, 16))
        self.assertEqual(skip.shape, (1, 8, 16))


if __name__ == "__main__":
    unittest.main()
