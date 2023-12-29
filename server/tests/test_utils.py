import unittest

from server.utils import sentences_split


class TestUtils(unittest.TestCase):
    def test_sentences_split(self):
        # Simple case 
        text = "Hello, world! This is a test sentence."
        sentences = sentences_split(text, 25)

        self.assertListEqual(
            sentences,
            [
                "Hello, world!",
                "This is a test sentence.",
            ],
        )

        # Case when the sentence is longer than `max_symbols`
        text = "This is a sentence. This is another sentence. Yet another sentence is here."

        self.assertRaises(ValueError, sentences_split, text, max_symbols=20)

        # Case when we have 2 sentences with the length of `max_symbols`
        sentences = sentences_split(text, max_symbols=50)

        self.assertListEqual(
            sentences,
            [
                "This is a sentence. This is another sentence.",
                "Yet another sentence is here.",
            ],
        )

        text = ""
        sentences = sentences_split(text, max_symbols=20)

        self.assertListEqual(sentences, [""])

        # text = "\"I'll email it. By the way,\" I turned to Riley, \"can I borrow your phone? I need to call my cell. I dropped it at the airport yesterday—\" I showed them the phone I'd found \"—and got this instead.\". I skipped the part where the owner was some hot Greek God. He was like Zeus and Hades combined."

        # sentences = sentences_split(text)
        # print(sentences)

