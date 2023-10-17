import unittest

from training.preprocess.tokenization import Tokenizer


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_tokenization(self):
        text = "This is a test sentence."
        expected_tokens = [101, 2025, 2005, 1039, 3233, 6253, 1014, 102]
        tokens = self.tokenizer(text)
        self.assertEqual(tokens, expected_tokens)

    def test_decode(self):
        tokens = [101, 2025, 2005, 1039, 3233, 6253, 1014, 102]
        expected_text = "[CLS] this is a test sentence. [SEP]"
        text = self.tokenizer.decode(tokens)
        self.assertEqual(text, expected_text)

    def test_tokenization_with_vocab_file(self):
        text = "This is a test sentence."
        vocab_file = "config/vocab_phonemes.txt"
        tokenizer = Tokenizer(vocab_file=vocab_file)
        expected_tokens = [64, 2, 2, 2, 2, 2, 2, 63]
        tokens = tokenizer(text)
        self.assertEqual(tokens, expected_tokens)

    def test_decode_without_special_tokens(self):
        tokens = [101, 2025, 2005, 1039, 3233, 6253, 1014, 102]
        expected_text = "this is a test sentence."
        text = self.tokenizer.decode(tokens)
        self.assertEqual(text, expected_text)

    def test_decode_with_special_tokens(self):
        tokens = [101, 2025, 2005, 1039, 3233, 6253, 1014, 102]
        expected_text = "[CLS] this is a test sentence. [SEP]"
        text = self.tokenizer.decode(tokens, False)
        self.assertEqual(text, expected_text)

    def test_encode_decode_phonemes(self):
        text = ['[SILENCE]', 'ð', 'ʌ', '[SILENCE]', 'd', 'eɪ', '[SILENCE]', 'æ', 'f', 't', 'ɜ˞', '[COMMA]', 'd', 'aɪ', 'æ', 'n', 'ʌ', '[SILENCE]', 'æ', 'n', 'd', '[SILENCE]', 'm', 'ɛ', 'ɹ', 'i', '[SILENCE]', 'k', 'w', 'ɪ', 't', 'ɪ', 'd', '[SILENCE]', 'ɪ', 't', '[SILENCE]', 'f', 'ɜ˞', '[SILENCE]', 'd', 'ɪ', 's', 't', 'ʌ', 'n', 't', '[SILENCE]', 'b', 'i', '[FULL STOP]']
        
        expected_tokens = [101, 104, 1100, 1136, 104, 1042, 30531, 104, 1099, 1044, 1058, 30535, 30524, 1042, 30528, 1099, 1052, 1136, 104, 1099, 1052, 1042, 104, 1051, 1117, 1127, 1047, 104, 1049, 1061, 1121, 1058, 1121, 1042, 104, 1121, 1058, 104, 1044, 30535, 104, 1042, 1121, 1057, 1058, 1136, 1052, 1058, 104, 1040, 1047, 105, 102]

        tokens = self.tokenizer(text)
        self.assertEqual(tokens, expected_tokens)

        expected_decoded_text = '[CLS] [SILENCE] ð ʌ [SILENCE] d eɪ [SILENCE] æ f t ɜ˞ [COMMA] d aɪ æ n ʌ [SILENCE] æ n d [SILENCE] m ɛ ɹ i [SILENCE] k w ɪ t ɪ d [SILENCE] ɪ t [SILENCE] f ɜ˞ [SILENCE] d ɪ s t ʌ n t [SILENCE] b i [FULL STOP] [SEP]'
        decoded_phonemes = self.tokenizer.decode(tokens)

        self.assertEqual(decoded_phonemes, expected_decoded_text)


if __name__ == "__main__":
    unittest.main()