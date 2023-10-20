import re

from nemo_text_processing.text_normalization.normalize import Normalizer
from unidecode import unidecode


class NormilizeText:
    r"""NVIDIA NeMo is a conversational AI toolkit built for researchers working on automatic speech recognition (ASR), text-to-speech synthesis (TTS), large language models (LLMs), and natural language processing (NLP). The primary objective of NeMo is to help researchers from industry and academia to reuse prior work (code and pretrained models) and make it easier to create new conversational AI models.

    This class normalize the characters in the input text and normalize the input text with the `nemo_text_processing`.

    Args:
        lang (str): The language code to use for normalization. Defaults to "en".

    Attributes:
        lang (str): The language code to use for normalization. Defaults to "en".
        model (Normalizer): The `nemo_text_processing` Normalizer model.

    Methods:
        byte_encode(word: str) -> list: Encode a word as a list of bytes.
        normalize_chars(text: str) -> str: Normalize the characters in the input text.
        __call__(text: str) -> str: Normalize the input text with the `nemo_text_processing`.

    Examples:
        >>> from training.preprocess.normilize_text import NormilizeText
        >>> normilize_text = NormilizeText()
        >>> normilize_text("It’s a beautiful day…")
        "It's a beautiful day..."
    """

    def __init__(self, lang: str = "en"):
        r"""Initialize a new instance of the NormilizeText class.

        Args:
            lang (str): The language code to use for normalization. Defaults to "en".

        """
        self.lang = lang

        self.model = Normalizer(input_case="cased", lang=lang)

    def byte_encode(self, word: str) -> list:
        r"""Encode a word as a list of bytes.

        Args:
            word (str): The word to encode.

        Returns:
            list: A list of bytes representing the encoded word.

        """
        text = word.strip()
        return list(text.encode("utf-8"))

    def normalize_chars(self, text: str) -> str:
        r"""Normalize the characters in the input text.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.

        Examples:
            >>> normalize_chars("It’s a beautiful day…")
            "It's a beautiful day..."

        """
        # Define the character mapping
        char_mapping = {
            ord("’"): ord("'"),
            ord("”"): ord('"'),
            ord("…"): ord("."),
            ord("„"): ord('"'),
            ord("“"): ord('"'),
            ord("–"): ord("-"),
            ord("«"): ord('"'),
            ord("»"): ord('"'),
        }

        # Normalize the characters using translate() method
        normalized_string = text.translate(char_mapping)

        # Add unicode normalization as additional garanty
        normalized_string = unidecode(normalized_string)

        # Remove redundant multiple characters
        # TODO: Maybe there is some effect on duplication?
        # re.sub(r"(\.|\!|\?)\1{2,}", r"\1\1\1", normalized_string)
        return re.sub(r"(\.|\!|\?|\-)\1+", r"\1",normalized_string)

    def __call__(self, text: str) -> str:
        r"""Normalize the input text with the `nemo_text_processing`.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.

        """
        result = self.normalize_chars(text)
        return self.model.normalize(result, punct_post_process=True)
