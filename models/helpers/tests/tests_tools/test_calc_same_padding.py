import unittest

from models.helpers.tools import calc_same_padding


class TestCalcSamePadding(unittest.TestCase):
    def test_odd_kernel_size(self):
        """Test that the padding returns correct tuple for odd-sized kernel."""
        kernel_size = 3  # an odd-sized kernel
        pad = calc_same_padding(kernel_size)
        self.assertEqual(pad, (1, 1))  # we expect padding of size 1 on both sides

    def test_even_kernel_size(self):
        """Test that the padding returns correct tuple for even-sized kernel."""
        kernel_size = 4  # an even-sized kernel
        pad = calc_same_padding(kernel_size)
        self.assertEqual(
            pad, (2, 1),
        )  # we expect padding of size 2 on one side, and size 1 on the other side

    def test_one_kernel_size(self):
        """Test that the padding returns correct tuple for kernel of size 1."""
        kernel_size = 1  # kernel of size 1
        pad = calc_same_padding(kernel_size)
        self.assertEqual(pad, (0, 0))  # we expect no padding on both sides

    def test_zero_kernel_size(self):
        """Test that the padding raises ValueError for invalid kernel size of zero."""
        with self.assertRaises(ValueError):
            calc_same_padding(0)  # a kernel of size 0 is not valid


if __name__ == "__main__":
    unittest.main()
