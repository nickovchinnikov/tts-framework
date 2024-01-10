from dataclasses import dataclass
from typing import Dict

from model.config import PreprocessLangType

# TODO: now we only support english, but we need to support other languages!
SUPPORTED_LANGUAGES = [
    "bg",
    "cs",
    "de",
    "en",
    "es",
    "fr",
    "ha",
    "hr",
    "ko",
    "pl",
    "pt",
    "ru",
    "sv",
    "sw",
    "th",
    "tr",
    "uk",
    "vi",
    "zh",
]

# Mappings from symbol to numeric ID and vice versa:
lang2id = {s: i for i, s in enumerate(SUPPORTED_LANGUAGES)}
id2lang = dict(enumerate(SUPPORTED_LANGUAGES))

@dataclass
class LangItem:
    r"""A class for storing language information."""

    phonemizer: str
    phonemizer_espeak: str
    nemo: str
    processing_lang_type: PreprocessLangType

langs_map: Dict[str, LangItem] = {
    "en": LangItem(
        phonemizer="en_us",
        phonemizer_espeak="en-us",
        nemo="en",
        processing_lang_type="english_only"
    ),
}

def get_lang_map(lang: str) -> LangItem:
    r"""Returns a LangItem object for the given language.

    Args:
        lang (str): The language to get the LangItem for.

    Raises:
        ValueError: If the language is not supported.

    Returns:
        LangItem: The LangItem object for the given language.
    """
    if lang not in langs_map:
        raise ValueError(f"Language {lang} is not supported!")
    return langs_map[lang]
