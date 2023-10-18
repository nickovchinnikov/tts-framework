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
id2lang = {i: s for i, s in enumerate(SUPPORTED_LANGUAGES)}

langs_map = {
    "en": {
        "phonemizer": "en_us",
        "nemo": "en",
        "processing_lang_type": "english_only",
    }
}
