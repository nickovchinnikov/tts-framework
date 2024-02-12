import unittest

import torch

from models.helpers import (
    stride_lens_downsampling,
)


class TestStrideLens(unittest.TestCase):
    def test_stride_lens(self):
        # Define test case inputs
        input_lengths = torch.tensor([5, 7, 10, 12])
        stride = 2

        # Correct output for this would be ceil([5, 7, 10, 12] / 2) => [3, 4, 5, 6]
        expected_output = torch.tensor([3, 4, 5, 6])

        # Call the function with the test cases
        output = stride_lens_downsampling(input_lengths, stride)

        # Check if the output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Check if the output shape is as expected
        self.assertEqual(output.shape, expected_output.shape)

        # Check if the output values are as expected
        self.assertTrue(torch.all(output.eq(expected_output)))

    def test_stride_lens_default_stride(self):
        # Define test case inputs. Here, we do not provide the stride.
        input_lengths = torch.tensor([10, 20, 4, 11])

        # Correct output for this would be ceil([10, 20, 4, 11] / 2) => [5, 10, 2, 6]
        expected_output = torch.tensor([5, 10, 2, 6])

        # Call the function with the test cases
        output = stride_lens_downsampling(input_lengths)

        # Check if the output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Check if the output shape is as expected
        self.assertEqual(output.shape, expected_output.shape)

        # Check if the output values are as expected
        self.assertTrue(torch.all(output.eq(expected_output)))


if __name__ == "__main__":
    unittest.main()
