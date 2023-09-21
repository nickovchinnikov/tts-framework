import unittest

import numpy as np
from numba import prange

from model.acoustic_model.mas import mas_width1, b_mas


class TestMasWidth1(unittest.TestCase):
    def setUp(self):
        # Example attn_map with random number between 0 and 1
        self.attn_map = np.random.rand(5, 5)
    
    def test_mas_width1(self):
        # Test output of mas_width1 function
        opt = mas_width1(self.attn_map)
        
        # Assert opt returned is a numpy ndarray
        self.assertIsInstance(opt, np.ndarray)

        # Assert the shapes of input attn_map and output opt are same
        self.assertEqual(opt.shape, self.attn_map.shape)

        # Assert opt only contains 0s and 1s (as per function description)
        self.assertTrue(np.array_equal(opt, opt.astype(bool)))
        
        # Assert that at least one entry in opt is 1.0 (since at least one optimal position should exist)
        self.assertIn(1.0, opt)



class TestBMas(unittest.TestCase):
    def setUp(self):
        # Create a sample batched attention map for testing
        # This sets up a batch of 2 attention maps, each of size 5 by 5
        self.b_attn_map = np.random.rand(2, 1, 5, 5)
        # Lengths of sequences in the input batch
        self.in_lens = np.array([3, 4])
        # Lengths of sequences in the output batch
        self.out_lens = np.array([4, 5])

    def test_b_mas(self):
        # Run the b_mas function taking width = 1
        attn_out = b_mas(self.b_attn_map, self.in_lens, self.out_lens, width=1)
 
        # Check the output type
        self.assertIsInstance(attn_out, np.ndarray)

        # Check if output and input have same shape
        self.assertEqual(attn_out.shape, self.b_attn_map.shape)

        # Assert attn_out only contains 0s and 1s.
        self.assertTrue(np.array_equal(attn_out, attn_out.astype(bool)))

        # Verify that the first dimension size equals batch size (2)
        self.assertEqual(attn_out.shape[0], 2)

        # Verify that the third and fourth dimensions size matches the given out_lens and in_len
        for b in prange(attn_out.shape[0]):
            self.assertEqual(np.sum(attn_out[b, 0, : self.out_lens[b], : self.in_lens[b]]), self.out_lens[b])


if __name__ == "__main__":
    unittest.main()
