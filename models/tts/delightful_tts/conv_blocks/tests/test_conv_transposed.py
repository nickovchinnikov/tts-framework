import unittest

import torch

from models.tts.delightful_tts.conv_blocks.bsconv import BSConv1d
from models.tts.delightful_tts.conv_blocks.conv_transposed import ConvTransposed


class TestConvTransposed(unittest.TestCase):
    def test_initialization(self):
        """Test to check if the ConvTransposed instance is properly created."""
        C_in, C_out, kernel_size, padding = 4, 6, 3, 1

        # Initialize ConvTransposed with input channels, output channels, kernel size and padding.
        conv_transposed = ConvTransposed(
            C_in,
            C_out,
            kernel_size,
            padding,
        )

        # Check type and value of 'conv' attribute which is an instance of BSConv1d
        self.assertIsInstance(
            conv_transposed.conv,
            BSConv1d,
            msg="Expected conv attribute to be instance of BSConv1d Layer",
        )

        # 'in_channels' and 'out_channels' attributes of conv.pointwise layer
        self.assertEqual(
            conv_transposed.conv.pointwise.conv.in_channels,
            C_in,
            msg=f"Expected conv.pointwise.conv.in_channels to be {C_in}, got {conv_transposed.conv.pointwise.conv.in_channels}",
        )

        self.assertEqual(
            conv_transposed.conv.pointwise.conv.out_channels,
            C_out,
            msg=f"Expected conv.pointwise.conv.out_channels to be {C_out}, got {conv_transposed.conv.pointwise.conv.out_channels}",
        )

        # Configuration attributes of conv.depthwise layer
        self.assertEqual(
            conv_transposed.conv.depthwise.conv.kernel_size[0],
            kernel_size,
            msg=f"Expected conv.depthwise.conv.kernel_size[0] to be {kernel_size}, got {conv_transposed.conv.depthwise.conv.kernel_size[0]}",
        )

        self.assertEqual(
            conv_transposed.conv.depthwise.conv.padding[0],
            padding,
            msg=f"Expected conv.depthwise.conv.padding[0] to be {padding}, got {conv_transposed.conv.depthwise.conv.padding[0]}",
        )

    def test_shape(self):
        """Test to check if the ConvTransposed instance is properly created and if the output shape is as expected."""
        N, C_in, C_out, kernel_size, padding, length = 5, 4, 6, 3, 1, 10

        # Initialize ConvTransposed with input channels, output channels, kernel size and padding.
        conv_transposed = ConvTransposed(
            C_in,
            C_out,
            kernel_size,
            padding,
        )

        # Generate a random tensor of shape (batch_size, channel, length).
        x = torch.randn(
            N,
            length,
            C_in,
        )

        out = conv_transposed(x)

        # The output shape should be the same as the input shape, as the ConvTransposed should not
        # alter the input's dimensions.
        self.assertEqual(out.shape, (N, length, C_out))


if __name__ == "__main__":
    unittest.main()
