import unittest

import torch

from models.tts.delightful_tts.diffusion.layers import (
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


class TestLayers(unittest.TestCase):
    def test_mish(self):
        mish = Mish()
        x = torch.randn(10, 10)
        self.assertTrue(torch.is_tensor(mish(x)))

    def test_upsample(self):
        upsample = Upsample(3)
        x = torch.randn(1, 3, 64, 64)
        self.assertEqual(upsample(x).shape, (1, 3, 128, 128))

    def test_downsample(self):
        downsample = Downsample(3)
        x = torch.randn(1, 3, 64, 64)
        self.assertEqual(downsample(x).shape, (1, 3, 32, 32))

    def test_rezero(self):
        rezero = Rezero(torch.nn.Linear(10, 10))
        x = torch.randn(10, 10)
        self.assertTrue(torch.is_tensor(rezero(x)))

    def test_block(self):
        block = Block(4, 16)
        x = torch.randn(1, 4, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        self.assertEqual(block(x, mask).shape, (1, 16, 64, 64))

    def test_resnet_block(self):
        resnet_block = ResnetBlock(4, 16, 10)
        x = torch.randn(1, 4, 64, 64)
        mask = torch.ones(1, 1, 64, 64)
        time_emb = torch.randn(1, 10)
        self.assertEqual(resnet_block(x, mask, time_emb).shape, (1, 16, 64, 64))

    def test_linear_attention(self):
        linear_attention = LinearAttention(3)
        x = torch.randn(1, 3, 64, 64)
        self.assertEqual(linear_attention(x).shape, (1, 3, 64, 64))

    def test_residual(self):
        residual = Residual(torch.nn.Linear(10, 10))
        x = torch.randn(10, 10)
        self.assertTrue(torch.is_tensor(residual(x)))

    def test_sinusoidal_pos_emb(self):
        sinusoidal_pos_emb = SinusoidalPosEmb(10)
        x = torch.randn(10)
        self.assertEqual(sinusoidal_pos_emb(x).shape, (10, 10))

if __name__ == "__main__":
    unittest.main()
