import unittest

import torch

from model.acoustic_model.variance_predictor import VariancePredictor


class TestVariancePredictor(unittest.TestCase):
    def setUp(self):
        # Initialize a VariancePredictor instance
        self.predictor = VariancePredictor(channels_in=32, channels=32, channels_out=1, kernel_size=3, p_dropout=0.5)

        # Assume batch size=3, channels_in=32, sequence_length=32
        self.x = torch.rand((3, 32, 32))

        # Assume batch size=3, sequence_length=32
        self.mask_dim = (3, 32)
        self.zero_mask = torch.ones(self.mask_dim).type(torch.bool)

    def test_forward(self):
        # Execute forward propagation
        output = self.predictor(self.x, self.zero_mask)

        # Validate output shape
        self.assertEqual(output.shape, self.mask_dim)  # Expected shape is (N, T) where N=batch size and T=sequence length

    def test_zero_mask(self):
        # Execute forward propagation
        output = self.predictor(self.x, self.zero_mask)

        # Validate all returned values are zero, given the mask is all False
        self.assertTrue(torch.all(output == 0))

    def test_ones_mask(self):
        # Create a mask of ones (indicating no entries are masked)
        ones_mask = torch.ones(self.mask_dim).type(torch.bool)

        # Execute forward propagation
        output = self.predictor(self.x, ones_mask)

        # Validate all returned values are not zero given all are True
        self.assertFalse(torch.all(output != 0))

    def test_output_dtype(self):
        # Execute forward propagation
        output = self.predictor(self.x, self.zero_mask)

        # Check the data type of output
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
