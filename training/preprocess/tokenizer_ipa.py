from typing import Union

from dp.phonemizer import Phonemizer
from dp.preprocessing.text import SequenceTokenizer

from model.config import get_lang_map

class TokenizerIPA:
    r"""
    TokenizerIPA is a class for tokenizing International Phonetic Alphabet (IPA) phonemes.

    Attributes:
        lang (str): Language to be used. Default is "en".
        phonemizer_checkpoint (str): Path to the phonemizer checkpoint file.
        phonemizer (Phonemizer): Phonemizer object for converting text to phonemes.
        tokenizer (SequenceTokenizer): SequenceTokenizer object for tokenizing the phonemes.
    """

    def __init__(
            self,
            lang: str = "en",
            phonemizer_checkpoint: str = "checkpoints/en_us_cmudict_ipa_forward.pt"
    ):
        r"""
        Initializes TokenizerIPA with the given language and phonemizer checkpoint.

        Args:
            lang (str): The language to be used. Default is "en".
            phonemizer_checkpoint (str): The path to the phonemizer checkpoint file.
        """
        lang_map = get_lang_map(lang)
        self.lang = lang_map.phonemizer

        self.phonemizer = Phonemizer.from_checkpoint(phonemizer_checkpoint)

        phoneme_symbols = [
            # IPA symbols
            'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɹ', 'ɡ', 'ɪ', 'ʁ', 'ʃ', 'ʊ', 'ʌ', 'ʏ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̃', '̍', '̥', '̩', '̯', '͡', 'θ',
            # Punctuation
            '!', '?', ',', '.', '-', ':', ';', '"', "'", '(', ')'
        ]

        self.tokenizer = SequenceTokenizer(phoneme_symbols,
                                        languages=['de', 'en_us'],
                                        lowercase=True,
                                        char_repeats=1,
                                        append_start_end=True)

    def __call__(self, text: str) -> tuple[Union[str, list[str]], list[int]]:
        r"""
        Converts the input text to phonemes and tokenizes them.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            list: The tokenized phonemes.

        """
        
        phones_ipa = self.phonemizer(text, lang=self.lang)
        tokens = self.tokenizer(phones_ipa, language=self.lang)

        return phones_ipa, tokens
