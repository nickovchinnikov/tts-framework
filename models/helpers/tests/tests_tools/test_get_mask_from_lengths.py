import unittest

import torch

from models.helpers.tools import get_mask_from_lengths


class TestGetMaskFromLengths(unittest.TestCase):
    def setUp(self):
        # Test cases: [2, 3, 1, 4]
        self.input_lengths = torch.tensor([2, 3, 1, 4])

    def test_get_mask_from_lengths(self):
        expected_output = torch.tensor(
            [
                [False, False, True, True],
                [False, False, False, True],
                [False, True, True, True],
                [False, False, False, False],
            ],
        )

        # Call the function with the test cases
        output = get_mask_from_lengths(self.input_lengths)

        # Check if the output is a tensor
        self.assertIsInstance(output, torch.Tensor)

        # Check if the output shape is as expected
        self.assertEqual(output.shape, expected_output.shape)

        # Check if the output values are as expected
        self.assertTrue(torch.all(output.eq(expected_output)))


if __name__ == "__main__":
    unittest.main()
