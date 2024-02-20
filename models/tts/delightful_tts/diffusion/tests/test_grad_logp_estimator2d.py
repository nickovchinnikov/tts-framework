from typing import Optional
import unittest

import torch
from torch import Tensor

from models.tts.delightful_tts.diffusion.grad_logp_estimator2d import (
    GradLogPEstimator2d,
)
from models.tts.delightful_tts.diffusion.utils import sequence_mask


class TestGradLogPEstimator2d(unittest.TestCase):
    def setUp(self):
        self.module = GradLogPEstimator2d(dim=32)
        self.batch_size = 32
        self.sequence_length = 64
        self.x = torch.randn(1, self.batch_size, self.sequence_length)

        x_lengths = torch.LongTensor([self.x.shape[-1]])
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, self.x.size(2)), 1)
        self.mask = x_mask

        self.mu = torch.randn(1, self.batch_size, self.sequence_length)

        offset = 1e-6
        t = torch.rand(self.x.shape[0], dtype=self.x.dtype)
        self.t = torch.clamp(t, offset, 1.0 - offset)

        self.spk = None

    def test_forward(self):
        output = self.module(self.x, self.mask, self.mu, self.t, self.spk)
        expected_shape = (1, self.batch_size, self.sequence_length)
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
