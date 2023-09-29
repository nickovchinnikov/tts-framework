import torch
import unittest

from model.conv_blocks.conv1d import DepthWiseConv1d, PointwiseConv1d

from model.helpers.tools import get_device


class TestDepthwiseConv1d(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        # initialize parameters once and reuse them in multiple test cases
        self.in_channels, self.out_channels, self.kernel_size, self.padding = 2, 4, 3, 1

        self.depthwise_conv = DepthWiseConv1d(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.padding,
            device=self.device,
        )

        # Generate an input data
        self.x_rand = torch.randn(32, self.in_channels, 64, device=self.device)
        self.x_zero = torch.zeros(32, self.in_channels, 64, device=self.device)
        self.x_ones = torch.ones(32, self.in_channels, 64, device=self.device)

    def test_forward(self):
        out = self.depthwise_conv(self.x_rand)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 64))

    def test_non_random_input(self):
        out = self.depthwise_conv(self.x_ones)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 64))

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 64))

    def test_zero_input(self):
        out = self.depthwise_conv(self.x_zero)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 64))

    def test_weight_change(self):
        self.depthwise_conv.conv.weight.data.fill_(0.5)
        out_first = self.depthwise_conv(self.x_rand)

        self.depthwise_conv.conv.weight.data.fill_(1.0)
        out_second = self.depthwise_conv(self.x_rand)

        # Ensuring weight changes have an effect
        self.assertTrue(torch.any(out_first != out_second))


class TestPointwiseConv1d(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        # initialize parameters once and reuse them in multiple test cases
        self.in_channels, self.out_channels, self.stride, self.padding, self.bias = (
            2,
            4,
            1,
            1,
            True,
        )

        self.pointwise_conv = PointwiseConv1d(
            self.in_channels, self.out_channels, self.stride, self.padding, self.bias
        )

        # Generate an input data
        self.x_rand = torch.randn(32, self.in_channels, 64, device=self.device)
        self.x_zero = torch.zeros(32, self.in_channels, 64, device=self.device)
        self.x_ones = torch.ones(32, self.in_channels, 64, device=self.device)

    def test_forward(self):
        out = self.pointwise_conv(self.x_rand)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        # Padding of 1 means one column of zeroes got added both at the beginning and at the end,
        # that's why you have 64 (original) + 2 (padding) = 66
        self.assertEqual(out.shape, (32, self.out_channels, 66))

    def test_non_random_input(self):
        out = self.pointwise_conv(self.x_ones)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 66))

    def test_zero_input(self):
        out = self.pointwise_conv(self.x_zero)

        self.assertEqual(out.device.type, self.device.type)

        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (32, self.out_channels, 66))

    def test_weight_change(self):
        self.pointwise_conv.conv.weight.data.fill_(0.5)
        out_first = self.pointwise_conv(self.x_rand)

        self.pointwise_conv.conv.weight.data.fill_(1.0)
        out_second = self.pointwise_conv(self.x_rand)

        self.assertTrue(torch.any(out_first != out_second))

    def test_kernel_size(self):
        # Checking if the module can handle non-default kernel sizes.
        kernel_size = 2
        pointwise_conv = PointwiseConv1d(
            self.in_channels,
            self.out_channels,
            self.stride,
            self.padding,
            self.bias,
            kernel_size,
        )
        out = pointwise_conv(self.x_rand)

        # ((input_size - kernel_size + 2*padding)/stride ) + 1
        # ((64 - 2 + 2*1) / 1 ) + 1 = 65
        self.assertEqual(out.shape, (32, self.out_channels, 65))


if __name__ == "__main__":
    unittest.main()
