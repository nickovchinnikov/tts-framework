import unittest

import torch

from .acoustic import AcousticModel


class TestTrainAcousticModel(unittest.TestCase):
    def setUp(self):
        self.model = AcousticModel()

    def test_forward(self):
        x = torch.randn(1, 28 * 28)
        y = self.model(x)
        self.assertEqual(y.size(), torch.Size([1, 10]))
