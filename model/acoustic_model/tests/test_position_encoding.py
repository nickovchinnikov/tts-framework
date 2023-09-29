# Required Libraries
import unittest
import torch

from model.acoustic_model.helpers import positional_encoding

from model.helpers.tools import get_device


class TestPositionalEncoding(unittest.TestCase):
    def test_positional_encoding(self):
        # Test with d_model=128, length=10 and device type CPU
        d_model = 128
        length = 10
        device = get_device()
        result = positional_encoding(d_model, length, device)

        # Assert the device type
        self.assertEqual(result.device.type, device.type)

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
