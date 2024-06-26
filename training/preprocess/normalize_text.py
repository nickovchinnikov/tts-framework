import re

from nemo_text_processing.text_normalization.normalize import Normalizer
from unidecode import unidecode


class NormalizeText:
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
        >>> from training.preprocess.normilize_text import NormalizeText
        >>> normilize_text = NormalizeText()
        >>> normilize_text("It’s a beautiful day…")
        "It's a beautiful day."
    """

    def __init__(self, lang: str = "en"):
        r"""Initialize a new instance of the NormalizeText class.

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
            "It's a beautiful day."

        """
        # Define the character mapping
        char_mapping = {
            ord("’"): ord("'"),
            ord("”"): ord("'"),
            ord("…"): ord("."),
            ord("„"): ord("'"),
            ord("“"): ord("'"),
            ord('"'): ord("'"),
            ord("–"): ord("-"),
            ord("—"): ord("-"),
            ord("«"): ord("'"),
            ord("»"): ord("'"),
        }

        # Add unicode normalization as additional garanty and normalize the characters using translate() method
        normalized_string = unidecode(text).translate(char_mapping)

        # Remove redundant multiple characters
        # TODO: Maybe there is some effect on duplication?
        return re.sub(r"(\.|\!|\?|\-)\1+", r"\1", normalized_string)

    def __call__(self, text: str) -> str:
        r"""Normalize the input text with the `nemo_text_processing`.

        Args:
            text (str): The input text to normalize.

        Returns:
            str: The normalized text.

        """
        text = self.normalize_chars(text)
        # return self.model.normalize(text)

        # Split the text into lines
        lines = text.split("\n")
        normalized_lines = self.model.normalize_list(lines)

        # TODO: check this!
        # Join the normalized lines, replace \n with . and return the result
        result = ". ".join(normalized_lines)
        return result
