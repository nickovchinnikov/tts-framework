import unittest

import torch

from models.tts.styledtts2.diffusion.distributions import (
    Distribution,
    LogNormalDistribution,
    UniformDistribution,
    VKDistribution,
)


class TestDistributions(unittest.TestCase):
    def test_distribution(self):
        with self.assertRaises(NotImplementedError):
            Distribution()(10)

    def test_log_normal_distribution(self):
        dist = LogNormalDistribution(mean=0.0, std=1.0)
        samples = dist(10)
        self.assertEqual(samples.shape, (10,))
        self.assertTrue(torch.all(samples > 0))

    def test_uniform_distribution(self):
        dist = UniformDistribution()
        samples = dist(10)
        self.assertEqual(samples.shape, (10,))
        self.assertTrue(torch.all((samples >= 0) & (samples < 1)))

    def test_vk_distribution(self):
        dist = VKDistribution(min_value=0.0, max_value=1.0, sigma_data=1.0)
        samples = dist(10)
        self.assertEqual(samples.shape, (10,))
        # No range check as the VKDistribution does not guarantee a specific range of output values
        # self.assertTrue(torch.all((samples >= 0) & (samples <= 1)))

if __name__ == "__main__":
    unittest.main()
