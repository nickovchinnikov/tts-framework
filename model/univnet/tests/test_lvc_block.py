import unittest

import torch
from torch import nn

from model.univnet.lvc_block import LVCBlock


class TestLVCBlock(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.in_channels = 3
        self.cond_channels = 4
        self.stride = 2

        self.in_length = 65536
        self.kernel_length = 10
        self.cond_length = 256
        self.dilations = [1, 3, 9, 27]
        self.lReLU_slope = 0.2
        self.conv_kernel_size = 3
        self.cond_hop_length = 256
        self.kpnet_hidden_channels = 64
        self.kpnet_conv_size = 3
        self.kpnet_dropout = 0.0

        self.x = torch.randn(self.batch_size, self.in_channels, self.in_length)

        self.kernel = torch.randn(
            self.batch_size,
            self.cond_channels,
            self.cond_length,
        )

        self.lvc_block = LVCBlock(
            in_channels=self.in_channels,
            cond_channels=self.cond_channels,
            stride=self.stride,
            dilations=self.dilations,
            lReLU_slope=self.lReLU_slope,
            conv_kernel_size=self.conv_kernel_size,
            cond_hop_length=self.cond_hop_length,
            kpnet_hidden_channels=self.kpnet_hidden_channels,
            kpnet_conv_size=self.kpnet_conv_size,
            kpnet_dropout=self.kpnet_dropout,
        )

    def test_remove_weight_norm(self):
        self.lvc_block.remove_weight_norm()

        for _, module in self.lvc_block.named_modules():
            if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d)):
                self.assertFalse(hasattr(module, "weight_g"))
                self.assertFalse(hasattr(module, "weight_v"))

    def test_location_variable_convolution(self):
        kernel = torch.randn(
            self.batch_size,
            self.in_channels,
            2 * self.in_channels,
            self.conv_kernel_size,
            self.cond_length,
        )
        bias = torch.randn(self.batch_size, 2 * self.in_channels, self.cond_length)

        output = self.lvc_block.location_variable_convolution(
            x=self.x,
            kernel=kernel,
            bias=bias,
            dilation=1,
            hop_size=self.cond_hop_length,
        )

        self.assertEqual(
            output.shape, (self.batch_size, 2 * self.in_channels, self.in_length),
        )

    def test_forward(self):
        x = torch.randn(
            self.batch_size,
            self.in_channels,
            self.in_length // self.stride,
        )

        output = self.lvc_block(x, self.kernel)

        self.assertEqual(
            output.shape, (self.batch_size, self.in_channels, self.in_length),
        )
