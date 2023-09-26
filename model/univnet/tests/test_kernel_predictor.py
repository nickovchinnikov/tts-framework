import torch
import unittest

from model.univnet.kernel_predictor import KernelPredictor

from helpers.tools import get_device


class TestKernelPredictor(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.batch_size = 2
        self.cond_channels = 4
        self.conv_in_channels = 3
        self.conv_out_channels = 5
        self.conv_layers = 2
        self.conv_kernel_size = 3
        self.kpnet_hidden_channels = 64
        self.kpnet_conv_size = 3
        self.kpnet_dropout = 0.0
        self.lReLU_slope = 0.1

        self.model = KernelPredictor(
            self.cond_channels,
            self.conv_in_channels,
            self.conv_out_channels,
            self.conv_layers,
            self.conv_kernel_size,
            self.kpnet_hidden_channels,
            self.kpnet_conv_size,
            self.kpnet_dropout,
            self.lReLU_slope,
            device=self.device,
        )

    def test_forward(self):
        c = torch.randn(self.batch_size, self.cond_channels, 10, device=self.device)
        kernels, bias = self.model(c)

        # Assert the device
        self.assertEqual(kernels.device.type, self.device.type)
        self.assertEqual(bias.device.type, self.device.type)

        self.assertIsInstance(kernels, torch.Tensor)
        self.assertEqual(
            kernels.shape,
            (
                self.batch_size,
                self.conv_layers,
                self.conv_in_channels,
                self.conv_out_channels,
                self.conv_kernel_size,
                10,
            ),
        )

        self.assertIsInstance(bias, torch.Tensor)
        self.assertEqual(
            bias.shape, (self.batch_size, self.conv_layers, self.conv_out_channels, 10)
        )

    def test_remove_weight_norm(self):
        self.model.remove_weight_norm()

        for module in self.model.modules():
            if hasattr(module, "weight_g"):
                self.assertIsNone(module.weight_g)
                self.assertIsNone(module.weight_v)
