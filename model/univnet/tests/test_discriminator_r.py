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

    def test_forward(self):
        x = torch.randn(1, 1024, device=self.device)

        # Test the forward pass of the DiscriminatorR class
        fmap, output = self.model(x)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        self.assertEqual(len(fmap), 6)

        first_dim, second_dim = 32, 1

        init_p = 9

        def dim_3rd(p: int = init_p):
            return max(2**p + 1, 2**6 + 1)

        # Assert the shape of the feature maps
        for i, fmap in enumerate(fmap[:-1]):
            self.assertEqual(
                fmap.shape, torch.Size([first_dim, second_dim, dim_3rd(init_p - i)])
            )

        self.assertEqual(fmap[-1].shape, torch.Size([second_dim, 65]))

        self.assertEqual(output.shape, (1, 65))

    def test_spectrogram(self):
        x = torch.randn(4, 1, 16384, device=self.device)
        # Test the spectrogram function of the DiscriminatorR class
        mag = self.model.spectrogram(x)

        # Assert the device
        self.assertEqual(mag.device.type, self.device.type)

        self.assertEqual(mag.shape, (4, 513))


if __name__ == "__main__":
    unittest.main()
