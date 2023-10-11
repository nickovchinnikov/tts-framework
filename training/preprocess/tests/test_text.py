import unittest

import numpy as np

from training.preprocess.text import NormilizeText


# Create a class to test the ComputePitch class
class TestTextPreprocess(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.normalizer = NormilizeText()

    def test_byte_encode(self):
        # Test with a simple word
        word = "hello"
        expected_output = [104, 101, 108, 108, 111]
        self.assertTrue(self.normalizer.byte_encode(word) == expected_output)

        # Test with a word containing non-ASCII characters
        word = "héllo"
        expected_output = [104, 195, 169, 108, 108, 111]
        self.assertTrue(self.normalizer.byte_encode(word) == expected_output)

        # Test with an empty string
        word = ""
        expected_output = []
        self.assertTrue(self.normalizer.byte_encode(word) == expected_output)

    def test_normalize_chars(self):
        # Test case 1: Test basic character normalization
        input_text = "It’s a beautiful day…"
        expected_output = "It's a beautiful day."
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 2: Test character normalization with multiple dots
        input_text = "Hello..... world!!!!"
        expected_output = "Hello. world!"
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 3: Test character normalization with multiple exclamation marks
        input_text = "Wow!!!!! This is amazing?????"
        expected_output = "Wow! This is amazing?"
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 4: Test character normalization with multiple question marks
        input_text = "What????? I don't understand!????"
        expected_output = "What? I don't understand!?"
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 5: Test character normalization with multiple quotes
        input_text = 'He said, “I don’t know…”'
        expected_output = 'He said, "I don\'t know."'
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 6: Test character normalization with multiple dashes
        input_text = "This is a long--sentence"
        expected_output = "This is a long-sentence"
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

        # Test case 7: Test character normalization with mixed characters
        input_text = "It’s a beautiful day… What????? I don't understand!!!!!"
        expected_output = "It's a beautiful day. What? I don't understand!"
        self.assertEqual(self.normalizer.normalize_chars(input_text), expected_output)

    def test_normalize(self):
        # Test case 1: Test basic text normalization
        input_text = r"""It’s a beautiful day… Hello..... World!!!! Wow!!!!! This is amazing????? He said, “I don’t know…”. It’s a beautiful day… What????? I don't understand!!!!!"""

        expected_output = r"""It's a beautiful day. Hello. World! Wow! This is amazing? He said, "I don't know.". It's a beautiful day. What? I don't understand!"""
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 2: Test text normalization with multiple dots
        input_text = "Hello..... World!!!!"
        expected_output = "Hello. World!"
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 3: numbers
        input_text = "1234567890"
        expected_output = "one two three four five six seven eight nine zero"
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 4: Complicated case
        input_text = "Mr. Smith paid $111 in U.S.A. on Dec. 17th. We paid $123 for this desk."
        expected_output = r"""mister Smith paid one hundred and eleven dollars in USA on december seventeenth. We paid one hundred and twenty three dollars for this desk."""
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 5: Complicated case 2
        input_text = "St. Patrick’s Day, spend $123 for this desk."
        expected_output = r"""Saint Patrick's Day, spend one hundred and twenty three dollars for this desk."""
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 6: check Dunky bug
        input_text = "For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'."
        expected_output = r"""For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'."""
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)

        # Test case 7: Time and currency
        input_text = "The alarm went off at 10:00a.m. \nI received $123. It's 12:30pm. I paid $123.45 for this desk."
        expected_output = r"""The alarm went off at ten AM I received one hundred and twenty three dollars. It's twelve thirty PM. I paid one hundred and twenty three dollars forty five cents for this desk."""
        self.assertEqual(self.normalizer.normalize(input_text), expected_output)


if __name__ == '__main__':
    unittest.main()