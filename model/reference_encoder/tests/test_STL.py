import unittest
from unittest.mock import Mock

import torch

from model.reference_encoder.STL import STL


class TestSTL(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model_config = Mock()
        cls.model_config.encoder.n_hidden = 512
        cls.model_config.reference_encoder.token_num = 32

        cls.stl = STL(cls.model_config)
        cls.batch_size = 10
        cls.n_hidden = cls.model_config.encoder.n_hidden

        cls.x = torch.rand(cls.batch_size, cls.n_hidden // 2)

    def test_forward(self):
        output = self.stl(self.x)

        self.assertTrue(torch.is_tensor(output))

        # Validate the output size
        expected_shape = (self.batch_size, 1, self.stl.attention.num_units)
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
