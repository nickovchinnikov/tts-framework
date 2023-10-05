import unittest

import numpy as np

from training.preprocess.text import (
    byte_encode,
)


# Create a class to test the ComputePitch class
class TestTextPreprocess(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_byte_encode(self):
        # Test with a simple word
        word = "hello"
        expected_output = [104, 101, 108, 108, 111]
        self.assertTrue(byte_encode(word) == expected_output)

        # Test with a word containing non-ASCII characters
        word = "héllo"
        expected_output = [104, 195, 169, 108, 108, 111]
        self.assertTrue(byte_encode(word) == expected_output)

        # Test with an empty string
        word = ""
        expected_output = []
        self.assertTrue(byte_encode(word) == expected_output)
