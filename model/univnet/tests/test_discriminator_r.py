import torch
import unittest

from helpers.tools import get_device

from model.univnet.discriminator_r import DiscriminatorR
from config import VocoderModelConfig


class TestDiscriminatorR(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.resolution = (1024, 256, 1024)
        self.model_config = VocoderModelConfig()
        self.model = DiscriminatorR(self.resolution, self.model_config, self.device)

        self.x = torch.randn(4, 1, 16384, device=self.device)

    def test_forward(self):
        x = torch.randn(1, 1024, device=self.device)

        # Test the forward pass of the DiscriminatorR class
        fmap, output = self.model(x)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        self.assertEqual(len(fmap), 6)
        self.assertEqual(output.shape, (1, 65))

    def test_spectrogram(self):
        # Test the spectrogram function of the DiscriminatorR class
        mag = self.model.spectrogram(self.x)

        # Assert the device
        self.assertEqual(mag.device.type, self.device.type)

        self.assertEqual(mag.shape, (4, 513))


if __name__ == "__main__":
    unittest.main()
