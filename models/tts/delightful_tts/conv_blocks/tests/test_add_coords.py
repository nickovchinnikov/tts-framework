import unittest

import torch

from models.tts.delightful_tts.conv_blocks.add_coords import AddCoords


# Test case for the AddCoords class
class TestAddCoords(unittest.TestCase):
    """Test case for the AddCoords class"""

    def test_rank_1_without_r(self):
        """Test for rank=1, with_r=False"""
        layer = AddCoords(rank=1, with_r=False)
        x = torch.rand(4, 3, 10)
        out = layer(x)
        self.assertEqual(list(out.shape), [4, 4, 10])

    def test_rank_1_with_r(self):
        """Test for rank=1, with_r=True"""
        layer = AddCoords(rank=1, with_r=True)
        x = torch.rand(4, 3, 10)
        out = layer(x)
        self.assertEqual(list(out.shape), [4, 5, 10])

    def test_rank_2_without_r(self):
        """Test for rank=2, with_r=False"""
        layer = AddCoords(rank=2, with_r=False)
        x = torch.rand(4, 3, 10, 20)
        out = layer(x)
        self.assertEqual(list(out.shape), [4, 5, 10, 20])

    def test_rank_2_with_r(self):
        """Test for rank=2, with_r=True"""
        layer = AddCoords(rank=2, with_r=True)
        x = torch.rand(4, 3, 10, 20)
        out = layer(x)
        self.assertEqual(list(out.shape), [4, 6, 10, 20])

    def test_rank_3_without_r(self):
        """Test for rank=3, with_r=False"""
        layer = AddCoords(rank=3, with_r=False)
        x = torch.rand(1, 3, 10, 20, 30)
        out = layer(x)
        self.assertEqual(list(out.shape), [1, 6, 10, 20, 30])

    def test_rank_3_with_r(self):
        """Test for rank=3, with_r=True"""
        layer = AddCoords(rank=3, with_r=True)
        x = torch.rand(1, 3, 10, 20, 30)
        out = layer(x)
        self.assertEqual(list(out.shape), [1, 7, 10, 20, 30])

    def test_not_implemented(self):
        """Test for not implemented rank."""
        layer = AddCoords(rank=4, with_r=False)
        x = torch.rand(4, 3, 10, 20, 30, 40)
        self.assertRaises(NotImplementedError, layer, x)


# Execute the unit test
if __name__ == "__main__":
    unittest.main()
