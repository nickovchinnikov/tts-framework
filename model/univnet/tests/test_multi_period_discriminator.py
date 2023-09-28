import torch
import unittest

import math

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

        fmaps_dims = [
            [
                torch.Size([self.batch_size, 64, 17, self.model_config.mpd.periods[0]]),
                torch.Size([self.batch_size, 128, 6, self.model_config.mpd.periods[0]]),
                torch.Size([self.batch_size, 256, 2, self.model_config.mpd.periods[0]]),
                torch.Size([self.batch_size, 512, 1, self.model_config.mpd.periods[0]]),
                torch.Size(
                    [self.batch_size, 1024, 1, self.model_config.mpd.periods[0]]
                ),
            ],
            [
                torch.Size([self.batch_size, 64, 12, self.model_config.mpd.periods[1]]),
                torch.Size([self.batch_size, 128, 4, self.model_config.mpd.periods[1]]),
                torch.Size([self.batch_size, 256, 2, self.model_config.mpd.periods[1]]),
                torch.Size([self.batch_size, 512, 1, self.model_config.mpd.periods[1]]),
                torch.Size(
                    [self.batch_size, 1024, 1, self.model_config.mpd.periods[1]]
                ),
            ],
            [
                torch.Size([self.batch_size, 64, 7, self.model_config.mpd.periods[2]]),
                torch.Size([self.batch_size, 128, 3, self.model_config.mpd.periods[2]]),
                torch.Size([self.batch_size, 256, 1, self.model_config.mpd.periods[2]]),
                torch.Size([self.batch_size, 512, 1, self.model_config.mpd.periods[2]]),
                torch.Size(
                    [self.batch_size, 1024, 1, self.model_config.mpd.periods[2]]
                ),
            ],
            [
                torch.Size([self.batch_size, 64, 5, self.model_config.mpd.periods[3]]),
                torch.Size([self.batch_size, 128, 2, self.model_config.mpd.periods[3]]),
                torch.Size([self.batch_size, 256, 1, self.model_config.mpd.periods[3]]),
                torch.Size([self.batch_size, 512, 1, self.model_config.mpd.periods[3]]),
                torch.Size(
                    [self.batch_size, 1024, 1, self.model_config.mpd.periods[3]]
                ),
            ],
            [
                torch.Size([self.batch_size, 64, 4, self.model_config.mpd.periods[4]]),
                torch.Size([self.batch_size, 128, 2, self.model_config.mpd.periods[4]]),
                torch.Size([self.batch_size, 256, 1, self.model_config.mpd.periods[4]]),
                torch.Size([self.batch_size, 512, 1, self.model_config.mpd.periods[4]]),
                torch.Size(
                    [self.batch_size, 1024, 1, self.model_config.mpd.periods[4]]
                ),
            ],
        ]

        init_2nd_dims = [17, 12, 7, 5, 4]

        for mpd_k in range(len(self.model_config.mpd.periods)):
            fmap = output[mpd_k][0]
            x = output[mpd_k][1]

            # Assert the device
            self.assertEqual(x.device.type, self.device.type)

            self.assertEqual(len(x), self.batch_size)

            # Assert the shape of the feature maps
            dim_2nd = init_2nd_dims[mpd_k]
            period = self.model_config.mpd.periods[mpd_k]

            dims_expl = fmaps_dims[mpd_k]

            for i in range(len(self.model_config.mpd.periods)):
                # Assert the device
                self.assertEqual(fmap[i].device.type, self.device.type)

                # Assert the shape of the feature maps explicitly
                self.assertEqual(fmap[i].shape, dims_expl[i])

                # Assert the shape of the feature maps
                self.assertEqual(
                    fmap[i].shape,
                    torch.Size([self.batch_size, 2 ** (i + 6), dim_2nd, period]),
                )
                dim_2nd = math.ceil(dim_2nd / self.model_config.mpd.stride)

        self.assertEqual(len(output), len(self.model_config.mpd.periods))


if __name__ == "__main__":
    unittest.main()
