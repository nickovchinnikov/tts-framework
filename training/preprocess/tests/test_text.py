import unittest

import numpy as np

from training.preprocess.text import byte_encode, normalize_chars


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

    def test_normalize_chars(self):
        # Test case 1: Test basic character normalization
        input_text = "It’s a beautiful day…"
        expected_output = "It's a beautiful day."
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 2: Test character normalization with multiple dots
        input_text = "Hello..... world!!!!"
        expected_output = "Hello. world!"
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 3: Test character normalization with multiple exclamation marks
        input_text = "Wow!!!!! This is amazing?????"
        expected_output = "Wow! This is amazing?"
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 4: Test character normalization with multiple question marks
        input_text = "What????? I don't understand!????"
        expected_output = "What? I don't understand!?"
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 5: Test character normalization with multiple quotes
        input_text = 'He said, “I don’t know…”'
        expected_output = 'He said, "I don\'t know."'
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 6: Test character normalization with multiple dashes
        input_text = "This is a long--sentence"
        expected_output = "This is a long-sentence"
        self.assertEqual(normalize_chars(input_text), expected_output)

        # Test case 7: Test character normalization with mixed characters
        input_text = "It’s a beautiful day… What????? I don't understand!!!!!"
        expected_output = "It's a beautiful day. What? I don't understand!"
        self.assertEqual(normalize_chars(input_text), expected_output)


if __name__ == '__main__':
    unittest.main()