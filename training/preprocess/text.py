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
