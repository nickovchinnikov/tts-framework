import unittest
from unittest.mock import Mock

import torch

from model.reference_encoder.STL import STL

from model.helpers.tools import get_device


class TestSTL(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.model_config = Mock()
        self.model_config.encoder.n_hidden = 512
        self.model_config.reference_encoder.token_num = 32

        self.stl = STL(self.model_config, device=self.device)
        self.batch_size = 10
        self.n_hidden = self.model_config.encoder.n_hidden

        self.x = torch.rand(self.batch_size, self.n_hidden // 2, device=self.device)

    def test_forward(self):
        output = self.stl(self.x)

        # Assert device type
        self.assertEqual(output.device.type, self.device.type)

        self.assertTrue(torch.is_tensor(output))

        # Validate the output size
        expected_shape = (self.batch_size, 1, self.stl.attention.num_units)
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
