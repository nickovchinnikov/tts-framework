import math
import unittest

import torch

from models.config import VocoderModelConfig
from models.vocoder.univnet import DiscriminatorP


class TestDiscriminatorP(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.channels = 1
        self.time_steps = 100
        self.period = 10
        self.model_config = VocoderModelConfig()

        self.x = torch.randn(self.batch_size, self.channels, self.time_steps)
        self.model = DiscriminatorP(self.period, self.model_config)

    def test_forward(self):
        fmap, output = self.model(self.x)

        self.assertEqual(len(fmap), len(self.model.convs) + 1)

        # Assert the shape of the feature maps explicitly
        fmap_dims = [
            torch.Size([self.batch_size, 64, 4, self.period]),
            torch.Size([self.batch_size, 128, 2, self.period]),
            torch.Size([self.batch_size, 256, 1, self.period]),
            torch.Size([self.batch_size, 512, 1, self.period]),
            torch.Size([self.batch_size, 1024, 1, self.period]),
            torch.Size([self.batch_size, 1, 1, self.period]),
        ]

        for i in range(len(fmap)):
            self.assertEqual(fmap[i].shape, fmap_dims[i])

        # Assert the shape of the feature maps
        dim_2nd = 4
        for i in range(len(self.model_config.mpd.periods)):
            self.assertEqual(fmap[i].shape[0], self.batch_size)
            self.assertEqual(fmap[i].shape[1], 2 ** (i + 6))
            self.assertEqual(fmap[i].shape[2], dim_2nd)

            dim_2nd = math.ceil(dim_2nd / self.model_config.mpd.stride)

            self.assertEqual(fmap[i].shape[3], self.period)

        self.assertEqual(output.shape, (self.batch_size, self.period))

    def test_forward_with_padding(self):
        fmap, output = self.model(self.x)

        self.assertEqual(len(fmap), len(self.model.convs) + 1)

        # Assert the shape of the feature maps explicitly
        fmap_dims = [
            torch.Size([self.batch_size, 64, 4, self.period]),
            torch.Size([self.batch_size, 128, 2, self.period]),
            torch.Size([self.batch_size, 256, 1, self.period]),
            torch.Size([self.batch_size, 512, 1, self.period]),
            torch.Size([self.batch_size, 1024, 1, self.period]),
            torch.Size([self.batch_size, 1, 1, self.period]),
        ]

        for i in range(len(fmap)):
            self.assertEqual(fmap[i].shape, fmap_dims[i])

        # Assert the shape of the feature maps
        dim_2nd = 4
        for i in range(len(self.model_config.mpd.periods)):
            self.assertEqual(fmap[i].shape[0], self.batch_size)
            self.assertEqual(fmap[i].shape[1], 2 ** (i + 6))
            self.assertEqual(fmap[i].shape[2], dim_2nd)

            dim_2nd = math.ceil(dim_2nd / self.model_config.mpd.stride)

            self.assertEqual(fmap[i].shape[3], self.period)

        self.assertEqual(output.shape, (self.batch_size, self.period))

    def test_forward_with_different_period(self):
        model = DiscriminatorP(self.period, self.model_config)
        x = torch.randn(self.batch_size, self.channels, self.time_steps - 1)

        model.period = 5
        fmap, output = model(x)

        self.assertEqual(len(fmap), len(model.convs) + 1)

        # Assert the shape of the feature maps explicitly
        fmap_dims = [
            torch.Size([self.batch_size, 64, 7, model.period]),
            torch.Size([self.batch_size, 128, 3, model.period]),
            torch.Size([self.batch_size, 256, 1, model.period]),
            torch.Size([self.batch_size, 512, 1, model.period]),
            torch.Size([self.batch_size, 1024, 1, model.period]),
            torch.Size([self.batch_size, 1, 1, model.period]),
        ]

        for i in range(len(fmap)):
            self.assertEqual(fmap[i].shape, fmap_dims[i])

        # Assert the shape of the feature maps
        dim_2nd = 7
        for i in range(len(self.model_config.mpd.periods)):
            self.assertEqual(fmap[i].shape[0], self.batch_size)
            self.assertEqual(fmap[i].shape[1], 2 ** (i + 6))
            self.assertEqual(fmap[i].shape[2], dim_2nd)

            dim_2nd = math.ceil(dim_2nd / self.model_config.mpd.stride)

            self.assertEqual(fmap[i].shape[3], model.period)

        self.assertEqual(output.shape, (self.batch_size, model.period))


if __name__ == "__main__":
    unittest.main()
