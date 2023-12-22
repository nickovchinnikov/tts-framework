import unittest

from training.preprocess.tokenizer_ipa import TokenizerIPA

class TestTokenizerIPA(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TokenizerIPA()
    
    def test_init(self):
        self.assertEqual(self.tokenizer.lang, "en_us")
        self.assertIsNotNone(self.tokenizer.phonemizer)
        self.assertIsNotNone(self.tokenizer.tokenizer)

    def test_call(self):
        text = "hello world"
        phones_ipa, tokens = self.tokenizer(text)

        self.assertIsInstance(phones_ipa, str)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens))

    def test_call_with_punctuation(self):
        text = "hello world"
        phones_ipa, tokens = self.tokenizer(text)

        text2 = "Hello, world!"
        phones_ipa2, tokens2 = self.tokenizer(text2)

        self.assertNotEqual(phones_ipa, phones_ipa2)
        self.assertNotEqual(tokens, tokens2)

if __name__ == '__main__':
    unittest.main()