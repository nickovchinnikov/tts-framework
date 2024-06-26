# NOTE: for the backward comp.
# Prepare the phonemes list and dictionary for the embedding
phoneme_basic_symbols = [
    # IPA symbols
    "a",
    "b",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "j",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y",
    "z",
    "æ",
    "ç",
    "ð",
    "ø",
    "ŋ",
    "œ",
    "ɐ",
    "ɑ",
    "ɔ",
    "ə",
    "ɛ",
    "ɝ",
    "ɹ",
    "ɡ",
    "ɪ",
    "ʁ",
    "ʃ",
    "ʊ",
    "ʌ",
    "ʏ",
    "ʒ",
    "ʔ",
    "ˈ",
    "ˌ",
    "ː",
    "̃",
    "̍",
    "̥",
    "̩",
    "̯",
    "͡",
    "θ",
    # Punctuation
    "!",
    "?",
    ",",
    ".",
    "-",
    ":",
    ";",
    '"',
    "'",
    "(",
    ")",
    " ",
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
phones = phoneme_basic_symbols + [
    symbol for symbol in symbols if symbol not in phoneme_basic_symbols
]

# TODO: Need to understand how to replace this
# len(phones) == 184, leave it as is at this point
symbols = [str(el) for el in range(256)]
symbol2id = {s: i for i, s in enumerate(symbols)}
id2symbol = {i: s for i, s in enumerate(symbols)}
