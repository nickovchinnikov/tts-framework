import unittest

import torch

from model.config import PreprocessingConfig, VocoderModelConfig
from model.univnet import Discriminator, UnivNet


# One of the most important test case for univnet
# Integration test
class TestDiscriminator(unittest.TestCase):
    def setUp(self):
        self.model_config = VocoderModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        self.generator = UnivNet(self.model_config, self.preprocess_config)

        self.model = Discriminator(self.model_config)

        self.batch_size = 1
        self.in_length = 100

        self.c = torch.randn(
            self.batch_size,
            self.preprocess_config.stft.n_mel_channels,
            self.in_length,
        )

    def test_forward(self):
        # Test the forward pass of the Discriminator class
        x = self.generator(self.c)

        output = self.model(x)

        self.assertEqual(len(output), 2)

        # Assert MRD length
        self.assertEqual(len(output[0]), 3)

        # Assert MPD length
        self.assertEqual(len(output[1]), 5)

        # Test MRD output
        # output_mrd = output[0]

        # fmap_mrd_dims = [
        #     [
        #         torch.Size([32, 1, 513]),
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([1, 1, 65]),
        #     ],
        #     [
        #         torch.Size([32, 1, 1025]),
        #         torch.Size([32, 1, 513]),
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([1, 1, 129]),
        #     ],
        #     [
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([32, 1, 33]),
        #         torch.Size([32, 1, 33]),
        #         torch.Size([1, 1, 33]),
        #     ],
        # ]

        # for key in range(len(output[0])):
        #     fmap = output_mrd[key][0]
        #     x = output_mrd[key][1]

        #     fmap_dims = fmap_mrd_dims[key]

        #     # Assert the shape of the feature maps
        #     for i, fmap_ in enumerate(fmap):
        #         # Assert the feature map shape explicitly
        #         self.assertEqual(fmap_.shape, fmap_dims[i])

        # # Test MPD output
        # output_mpd = output[1]

        # fmap_mpd_dims = [
        #     [
        #         torch.Size([1, 64, 4267, 2]),
        #         torch.Size([1, 128, 1423, 2]),
        #         torch.Size([1, 256, 475, 2]),
        #         torch.Size([1, 512, 159, 2]),
        #         torch.Size([1, 1024, 159, 2]),
        #         torch.Size([1, 1, 159, 2]),
        #     ],
        #     [
        #         torch.Size([1, 64, 2845, 3]),
        #         torch.Size([1, 128, 949, 3]),
        #         torch.Size([1, 256, 317, 3]),
        #         torch.Size([1, 512, 106, 3]),
        #         torch.Size([1, 1024, 106, 3]),
        #         torch.Size([1, 1, 106, 3]),
        #     ],
        #     [
        #         torch.Size([1, 64, 1707, 5]),
        #         torch.Size([1, 128, 569, 5]),
        #         torch.Size([1, 256, 190, 5]),
        #         torch.Size([1, 512, 64, 5]),
        #         torch.Size([1, 1024, 64, 5]),
        #         torch.Size([1, 1, 64, 5]),
        #     ],
        #     [
        #         torch.Size([1, 64, 1220, 7]),
        #         torch.Size([1, 128, 407, 7]),
        #         torch.Size([1, 256, 136, 7]),
        #         torch.Size([1, 512, 46, 7]),
        #         torch.Size([1, 1024, 46, 7]),
        #         torch.Size([1, 1, 46, 7]),
        #     ],
        #     [
        #         torch.Size([1, 64, 776, 11]),
        #         torch.Size([1, 128, 259, 11]),
        #         torch.Size([1, 256, 87, 11]),
        #         torch.Size([1, 512, 29, 11]),
        #         torch.Size([1, 1024, 29, 11]),
        #         torch.Size([1, 1, 29, 11]),
        #     ],
        # ]

        # for key in range(len(output[1])):
        #     fmap = output_mpd[key][0]
        #     x = output_mpd[key][1]

        #     fmap_dims = fmap_mpd_dims[key]

        #     # Assert the shape of the feature maps
        #     for i, fmap in enumerate(fmap):
        #         # Assert the feature map shape explicitly
        #         self.assertEqual(fmap.shape, fmap_dims[i])


if __name__ == "__main__":
    unittest.main()
