import torch
import unittest

from model.conv_blocks.coord_conv1d import CoordConv1d


class TestCoordConv1d(unittest.TestCase):
    def test_simple_case(self):
        """Tests a simple case with input of size (1, 2, 10)"""
        rank1_model = CoordConv1d(
            in_channels=2,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            with_r=False,
        )
        x = torch.randn(1, 2, 10)
        output = rank1_model(x)
        self.assertEqual(list(output.shape), [1, 8, 8])

    def test_with_r(self):
        """Tests if 'with_r' adds an extra radial channel"""
        model = CoordConv1d(
            in_channels=2,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            with_r=True,
        )
        x = torch.randn(1, 2, 10)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 8, 8])

    def test_with_padding(self):
        """Tests if padding is functioning correctly"""
        model = CoordConv1d(
            in_channels=2,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=True,
            with_r=False,
        )
        x = torch.randn(1, 2, 10)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 8, 10])


# To run the tests
if __name__ == "__main__":
    unittest.main()
