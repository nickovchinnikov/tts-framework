# Required Libraries
import unittest

import torch

from models.helpers import positional_encoding


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        # Test with d_model=128, length=10
        d_model = 128
        length = 10
        result = positional_encoding(d_model, length)

        # Assert that output is a torch.Tensor
        self.assertIsInstance(result, torch.Tensor)

        # Assert the output tensor shape is correct
        # The extra dimension from unsqueeze is considered
        expected_shape = (1, length, d_model)
        self.assertEqual(result.shape, expected_shape)

        # Assert that values lie in the range [-1, 1]
        self.assertTrue(torch.all((result >= -1) & (result <= 1)))


# Run tests
if __name__ == "__main__":
    unittest.main()
