import unittest

import torch

from models.config import VocoderModelConfig
from models.vocoder.univnet import MultiResolutionDiscriminator


class TestMultiResolutionDiscriminator(unittest.TestCase):
    def setUp(self):
        self.resolution = [(1024, 256, 1024), (2048, 512, 2048)]
        self.model_config = VocoderModelConfig()
        self.model = MultiResolutionDiscriminator(self.model_config)

        self.x = torch.randn(1, 1024)

    def test_forward(self):
        # Test the forward pass of the MultiResolutionDiscriminator class
        output = self.model(self.x)
        self.assertEqual(len(output), 3)

        # fmap_dims = [
        #     [
        #         torch.Size([32, 1, 513]),
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([1, 65]),
        #     ],
        #     [
        #         torch.Size([32, 1, 1025]),
        #         torch.Size([32, 1, 513]),
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([1, 129]),
        #     ],
        #     [
        #         torch.Size([32, 1, 257]),
        #         torch.Size([32, 1, 129]),
        #         torch.Size([32, 1, 65]),
        #         torch.Size([32, 1, 33]),
        #         torch.Size([32, 1, 33]),
        #         torch.Size([1, 33]),
        #     ],
        # ]

        # init_powers_max_min = [(9, 6), (10, 7), (8, 5)]

        # for key in range(len(output)):
        #     fmap = output[key][0]

        #     first_dim, second_dim = 32, 1

        #     p_max, p_min = init_powers_max_min[key]

        #     def dim_3rd(p: int):
        #         return max(2**p + 1, 2**p_min + 1)

        #     fmap_dim = fmap_dims[key]

        #     # Assert the shape of the feature maps
        #     for i, fmap_ in enumerate(fmap[:-1]):
        #         # Assert the feature map shape explicitly
        #         self.assertEqual(fmap_.shape, fmap_dim[i])

        #         self.assertEqual(
        #             fmap_.shape, torch.Size([first_dim, second_dim, dim_3rd(p_max - i)]),
        #         )

        #     self.assertEqual(fmap[-1].shape, torch.Size([second_dim, second_dim, 2**p_min + 1]))


if __name__ == "__main__":
    unittest.main()
