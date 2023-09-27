import torch
import unittest

from model.univnet import DiscriminatorP
from config import VocoderModelConfig

from helpers.tools import get_device


class TestDiscriminatorP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 1
        self.time_steps = 100
        self.period = 10
        self.model_config = VocoderModelConfig()
        self.device = get_device()

        self.x = torch.randn(
            self.batch_size, self.channels, self.time_steps, device=self.device
        )
        self.model = DiscriminatorP(self.period, self.model_config, self.device)

    def test_forward(self):
        fmap, output = self.model(self.x)

        # Assert the device
        self.assertEqual(self.model.device.type, self.device.type)

        self.assertEqual(len(fmap), len(self.model.convs) + 1)
        self.assertEqual(output.shape, (self.batch_size, self.period))

    def test_forward_with_padding(self):
        fmap, output = self.model(self.x)

        self.assertEqual(len(fmap), len(self.model.convs) + 1)
        self.assertEqual(output.shape, (self.batch_size, self.period))

    def test_forward_with_different_period(self):
        model = DiscriminatorP(self.period, self.model_config, self.device)
        x = torch.randn(
            self.batch_size, self.channels, self.time_steps - 1, device=self.device
        )

        model.period = 5
        fmap, output = model(x)

        self.assertEqual(len(fmap), len(model.convs) + 1)
        self.assertEqual(output.shape, (self.batch_size, model.period))


if __name__ == "__main__":
    unittest.main()
