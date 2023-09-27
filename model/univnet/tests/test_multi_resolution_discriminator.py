import torch
import unittest

from helpers.tools import get_device

from model.univnet import MultiResolutionDiscriminator
from config import VocoderModelConfig


class TestMultiResolutionDiscriminator(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.resolution = [(1024, 256, 1024), (2048, 512, 2048)]
        self.model_config = VocoderModelConfig()
        self.model = MultiResolutionDiscriminator(self.model_config, self.device)

        self.x = torch.randn(1, 1024, device=self.device)

    def test_forward(self):
        # Test the forward pass of the MultiResolutionDiscriminator class
        output = self.model(self.x)

        self.assertEqual(len(output), 3)

        # output shape is to unpredicatable to cover...


if __name__ == "__main__":
    unittest.main()
