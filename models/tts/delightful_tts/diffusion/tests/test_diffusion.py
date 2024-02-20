import unittest

import torch

from models.tts.delightful_tts.diffusion.diffusion import (
    Diffusion,
)
from models.tts.delightful_tts.diffusion.utils import sequence_mask


class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.diffusion = Diffusion(n_feats=80, dim=32)

        self.batch_size = 32
        self.sequence_length = 64

        self.x = torch.randn(1, self.batch_size, self.sequence_length)

        x_lengths = torch.LongTensor([self.x.shape[-1]])
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, self.x.size(2)), 1)
        self.mask = x_mask

        self.mu = torch.randn(1, self.batch_size, self.sequence_length)

        self.z = torch.randn(1, self.batch_size, self.sequence_length)

        offset = 1e-6
        t = torch.rand(self.x.shape[0], dtype=self.x.dtype)
        self.t = torch.clamp(t, offset, 1.0 - offset)

        self.n_timesteps = 10
        self.stoc = False
        self.spk = None

    def test_forward_diffusion(self):
        xt, noise = self.diffusion.forward_diffusion(self.z, self.mask, self.mu, self.t)
        # Check shapes
        self.assertEqual(xt.shape, (1, self.batch_size, self.sequence_length))
        self.assertEqual(noise.shape, (1, self.batch_size, self.sequence_length))

    def test_reverse_diffusion(self):
        xt_reverse = self.diffusion.reverse_diffusion(self.z, self.mask, self.mu, self.n_timesteps, self.stoc, self.spk)
        # Check shape
        self.assertEqual(xt_reverse.shape, (1, self.batch_size, self.sequence_length))

    def test_forward(self):
        output = self.diffusion.forward(self.z, self.mask, self.mu, self.n_timesteps, self.stoc, self.spk)
        # Check shape
        self.assertEqual(output.shape, (1, self.batch_size, self.sequence_length))

    def test_loss_t(self):
        loss, xt = self.diffusion.loss_t(self.z, self.mask, self.mu, self.t)  # Example time tensor
        # Check shapes
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(xt, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(xt.shape, (1, self.batch_size, self.sequence_length))

    def test_compute_loss(self):
        loss, xt = self.diffusion.compute_loss(self.z, self.mask, self.mu, self.spk)
        # Check shapes
        self.assertIsInstance(loss, torch.Tensor)
        self.assertIsInstance(xt, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertEqual(xt.shape, (1, self.batch_size, self.sequence_length))


if __name__ == "__main__":
    unittest.main()
