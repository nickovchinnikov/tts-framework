from typing import List, Union

from transformers import BertTokenizer


# EXPERIMENTAL CLASS
# NOTE: Tokenization is failed, because of some symbols in the text. I need to find out more sustanable way.
# Can't be sure about tokens for IPA. Dict is not a good idea, because of the same reason.
class Tokenizer:
    r"""A wrapper class for the BERT tokenizer from the Hugging Face Transformers library.
    Use this with `vocab_file` and it makes sure that the correct vocabulary is used.

    Args:
        checkpoint (str): The name or path of the pre-trained BERT checkpoint to use.
        vocab_file (str): The path to the custom vocabulary file to use (optional).

    Attributes:
        tokenizer (BertTokenizer): The BERT tokenizer object.

    """

    def __init__(self, checkpoint: str = "bert-base-uncased", vocab_file: str = "config/vocab.txt") -> None:
        r"""Initializes the Tokenizer object with the specified checkpoint and vocabulary file.

        Args:
            checkpoint (str): The name or path of the pre-trained BERT checkpoint to use.
            vocab_file (str): The path to the custom vocabulary file to use (optional).

        Returns:
            None.

        """
        self.tokenizer = BertTokenizer.from_pretrained(checkpoint, vocab_file=vocab_file)

    def __call__(self, text: Union[str, List[str]], add_special_tokens: bool = True) -> list[int]:
        r"""Tokenizes the input text using the BERT tokenizer.

        Args:
            text (str): The input text to tokenize.
            add_special_tokens (bool): Whether to add special tokens to the tokenized text (optional).

        Returns:
            tokens (List[int]): A list of token IDs representing the tokenized text.

        """
        tokens = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return tokens

    def decode(self, tokens: list[int], skip_special_tokens: bool = True) -> list[str]:
        r"""Decodes the input token IDs into a list of strings.

        Args:
            tokens (List[int]): A list of token IDs to decode.
            skip_special_tokens (bool): Whether to add special tokens to the tokenized text (optional).

        Returns:
            text (List[str]): A list of strings representing the decoded tokens.

        """
        text_list = self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
        return text_list
