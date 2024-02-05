from logging import ERROR, Logger
import os

from phonemizer.backend import EspeakBackend

# IPA Phonemizer: https://github.com/bootphon/phonemizer
from phonemizer.backend.espeak.wrapper import EspeakWrapper

# Create a Logger instance
logger = Logger("my_logger")
# Set the level to ERROR
logger.setLevel(ERROR)

from dp.preprocessing.text import SequenceTokenizer

from model.config import get_lang_map

# INFO: Fix for windows, used for local env
if os.name == "nt":
    ESPEAK_LIBRARY = os.getenv(
        "ESPEAK_LIBRARY",
        "C:\\Program Files\\eSpeak NG\\libespeak-ng.dll",
    )
    EspeakWrapper.set_library(ESPEAK_LIBRARY)


class TokenizerIpaEspeak:
    def __init__(self, lang: str = "en"):
        lang_map = get_lang_map(lang)
        self.lang = lang_map.phonemizer_espeak
        self.lang_seq = lang_map.phonemizer

        # NOTE: for the backward comp.
        # Prepare the phonemes list and dictionary for the embedding
        phoneme_basic_symbols = [
            # IPA symbols
            "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ç", "ð", "ø", "ŋ", "œ", "ɐ", "ɑ", "ɔ", "ə", "ɛ", "ɝ", "ɹ", "ɡ", "ɪ", "ʁ", "ʃ", "ʊ", "ʌ", "ʏ", "ʒ", "ʔ", "ˈ", "ˌ", "ː", "̃", "̍", "̥", "̩", "̯", "͡", "θ",
            # Punctuation
            "!", "?", ",", ".", "-", ":", ";", '"', "'", "(", ")", " ",
        ]

        # TODO: add support for other languages
        # _letters_accented = "µßàáâäåæçèéêëìíîïñòóôöùúûüąćęłńœśşźżƒ"
        # _letters_cyrilic = "абвгдежзийклмнопрстуфхцчшщъыьэюяёєіїґӧ"
        # _pad = "$"

        # This is the list of symbols from StyledTTS2
        _punctuation = ';:,.!?¡¿—…"«»“”'
        _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

        # Combine all symbols
        symbols = list(_punctuation) + list(_letters) + list(_letters_ipa)

        # Add only unique symbols
        phones = phoneme_basic_symbols + [symbol for symbol in symbols if symbol not in phoneme_basic_symbols]

        # NOTE: for backward compatibility with previous IPA tokenizer see the TokenizerIPA class
        self.tokenizer = SequenceTokenizer(phones,
                                           languages=["de", "en_us"],
                                           lowercase=True,
                                           char_repeats=1,
                                           append_start_end=True)

        self.phonemizer = EspeakBackend(
            language=self.lang,
            preserve_punctuation=True,
            with_stress=True,
            words_mismatch="ignore",
            logger=logger,
        ).phonemize

    def __call__(self, text: str):
        r"""Converts the input text to phonemes and tokenizes them.

        Args:
            text (str): The input text to be tokenized.

        Returns:
            Tuple[Union[str, List[str]], List[int]]: IPA phonemes and tokens.

        """
        phones_ipa = "".join(self.phonemizer([text]))

        tokens = self.tokenizer(phones_ipa, language=self.lang_seq)

        return phones_ipa, tokens
