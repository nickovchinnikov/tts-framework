import unittest
from unittest.mock import Mock

import torch

from models.tts.delightful_tts.reference_encoder.STL import STL


class TestSTL(unittest.TestCase):
    def setUp(self):
        self.model_config = Mock()
        self.model_config.encoder.n_hidden = 512
        self.model_config.reference_encoder.token_num = 32

        self.stl = STL(
            self.model_config,
        )
        self.batch_size = 10
        self.n_hidden = self.model_config.encoder.n_hidden

        self.x = torch.rand(
            self.batch_size,
            self.n_hidden // 2,
        )

    def test_forward(self):
        output = self.stl(self.x)

        self.assertTrue(torch.is_tensor(output))

        # Validate the output size
        expected_shape = (self.batch_size, 1, self.stl.attention.num_units)
        self.assertEqual(output.shape, expected_shape)


if __name__ == "__main__":
    unittest.main()
