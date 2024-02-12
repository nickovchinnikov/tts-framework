import unittest

import torch
from torch import nn

from models.tts.styledtts2.diffusion.diffusion import (
    Diffusion,
    KDiffusion,
    VDiffusion,
    VKDiffusion,
)
from models.tts.styledtts2.diffusion.distributions import Distribution

torch.manual_seed(0)

class TestDistributions(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([1.0])

    class MockNet(nn.Module):
        def forward(self, x, *args, **kwargs):
            return x

    class MockDistribution(Distribution):
        def __call__(self, num_samples: int):
            return torch.ones(num_samples)

    def test_diffusion(self):
        with self.assertRaises(NotImplementedError):
            Diffusion().denoise_fn(self.x)
        with self.assertRaises(NotImplementedError):
            Diffusion()(self.x)

    def test_v_diffusion(self):
        net = self.MockNet()
        dist = self.MockDistribution()

        v_diffusion = VDiffusion(net, sigma_distribution=dist)
        self.assertFalse(torch.allclose(v_diffusion(self.x), self.x))

    def test_k_diffusion(self):
        net = self.MockNet()
        dist = self.MockDistribution()

        k_diffusion = KDiffusion(net, sigma_distribution=dist, sigma_data=1.0)
        self.assertFalse(torch.allclose(k_diffusion(self.x), self.x))

    def test_vk_diffusion(self):
        net = self.MockNet()
        dist = self.MockDistribution()

        vk_diffusion = VKDiffusion(net, sigma_distribution=dist)
        self.assertFalse(torch.allclose(vk_diffusion(self.x), self.x))

if __name__ == "__main__":
    unittest.main()
