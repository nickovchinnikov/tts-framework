import re

from unidecode import unidecode


def byte_encode(word: str) -> list:
    r"""
    Encode a word as a list of bytes.

    Args:
        word (str): The word to encode.

    Returns:
        list: A list of bytes representing the encoded word.

    """
    text = word.strip()
    return list(text.encode("utf-8"))


def normalize_chars(text: str) -> str:
    r"""
    Normalize the characters in the input text.

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
    normalized_string = re.sub(r"(\.|\!|\?|\-)\1+", r"\1",normalized_string)
    # TODO: Maybe there is some effect on duplication?
    # re.sub(r"(\.|\!|\?)\1{2,}", r"\1\1\1", normalized_string)


    return normalized_string
