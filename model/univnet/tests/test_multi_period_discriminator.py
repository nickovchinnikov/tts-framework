import torch
import unittest

from model.univnet import MultiPeriodDiscriminator
from config import VocoderModelConfig

from helpers.tools import get_device


class TestMultiPeriodDiscriminator(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 1
        self.time_steps = 100
        self.model_config = VocoderModelConfig()
        self.device = get_device()

        self.model = MultiPeriodDiscriminator(self.model_config, self.device)
        self.x = torch.randn(
            self.batch_size, self.channels, self.time_steps, device=self.device
        )

    def test_forward(self):
        output = self.model(self.x)

        self.assertEqual(len(output), len(self.model_config.mpd.periods))

        for mpd_k in range(len(self.model_config.mpd.periods)):
            fmap = output[mpd_k][0]
            x = output[mpd_k][1]
            self.assertEqual(len(x), self.batch_size)

            for i, _ in enumerate(self.model_config.mpd.periods):
                self.assertEqual(fmap[i].shape[0], self.batch_size)
                self.assertEqual(fmap[i].shape[1], 2 ** (i + 6))

                # dim 2 and 3 are very complicated to calculate...

        self.assertEqual(len(output), len(self.model_config.mpd.periods))


if __name__ == "__main__":
    unittest.main()
