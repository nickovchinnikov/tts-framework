import unittest

import torch

from models.helpers.tools import pad


class TestPad(unittest.TestCase):
    def test_1D_tensors_pad(self):
        # 1D tensor inputs
        tensors = [torch.ones(n) for n in range(1, 11)]
        # Ten 1D tensors of length 1 to 10
        max_len = max(t.numel() for t in tensors)

        # Call the function
        result = pad(tensors, max_len)

        # Check the output shape is as expected
        self.assertTrue(all(t.numel() == max_len for t in result))

    def test_2D_tensors_pad(self):
        # 2D tensor inputs
        tensors = [torch.ones(n, 5) for n in range(1, 11)]
        max_len = max(t.size(0) for t in tensors)

        # Call the function
        result = pad(tensors, max_len)

        # Check the output shape is as expected
        self.assertTrue(all(t.size(0) == max_len for t in result))
        # Make sure second dimension wasn't changed
        self.assertTrue(all(t.size(1) == 5 for t in result))


if __name__ == "__main__":
    unittest.main()
