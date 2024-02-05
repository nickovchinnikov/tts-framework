import unittest

from training.preprocess.tokenizer_ipa import TokenizerIPA
from training.preprocess.tokenizer_ipa_espeak import TokenizerIpaEspeak


class TestTokenizerIPA(unittest.TestCase):
    def setUp(self):
        self.tokenizerIPA = TokenizerIPA()
        self.tokenizerIpaEspeak = TokenizerIpaEspeak()

    def test_init(self):
        self.assertEqual(self.tokenizerIPA.lang, "en_us")
        self.assertIsNotNone(self.tokenizerIPA.phonemizer)
        self.assertIsNotNone(self.tokenizerIPA.tokenizer)

    def test_call(self):
        text = "'DUDLEY! MR DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON'T BELIEVE WHAT IT'S DOING!'Dudley came waddling towards them as fast as he could."
        # phones_ipa, tokens = self.tokenizerIPA(text)
        phones_ipa_espeak, tokens_espeak = self.tokenizerIpaEspeak(text)

        self.assertIsInstance(phones_ipa_espeak, str)
        self.assertIsInstance(tokens_espeak, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens_espeak))

    def test_case_sensitive(self):
        text = "WHAT IT'S DOING! Dudley came waddling towards them as fast as he could."
        # phones_ipa, tokens = self.tokenizerIPA(text)
        phones_ipa_espeak, tokens_espeak = self.tokenizerIpaEspeak(text)

        self.assertIsInstance(phones_ipa_espeak, str)
        self.assertIsInstance(tokens_espeak, list)
        self.assertTrue(all(isinstance(token, int) for token in tokens_espeak))

    def test_call_with_punctuation(self):
        text = "hello world"
        phones_ipa, tokens = self.tokenizerIpaEspeak(text)

        text2 = "Hello, world!"
        phones_ipa2, tokens2 = self.tokenizerIpaEspeak(text2)

        self.assertNotEqual(phones_ipa, phones_ipa2)
        self.assertNotEqual(tokens, tokens2)

if __name__ == "__main__":
    unittest.main()
