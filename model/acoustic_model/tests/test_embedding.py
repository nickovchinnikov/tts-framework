import torch
import unittest

from model.acoustic_model.embedding import Embedding

from helpers.tools import get_device


class TestEmbedding(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        self.embedding = Embedding(
            num_embeddings=100, embedding_dim=50, device=self.device
        )

    def test_forward_output_shape(self):
        # Generate a tensor of indices to lookup in the embedding
        idx = torch.randint(
            low=0, high=100, size=(10, 20), device=self.device
        )  # for a sequence of length 20 and batch size 10

        # Test the forward function
        output = self.embedding(idx)

        # Assert the device type of output
        self.assertEqual(output.device.type, self.device.type)

        # Check the output's shape is as expected
        self.assertEqual(output.shape, (10, 20, 50))

    def test_forward_output_values(self):
        idx = torch.LongTensor([[0, 50], [99, 1]]).to(
            self.device
        )  # Indices to lookup in the embedding

        output = self.embedding(idx)

        # Check the values returned by forward function match the expected embeddings
        self.assertTrue(torch.all(output[0, 0] == self.embedding.embeddings[0]))
        self.assertTrue(torch.all(output[1, 1] == self.embedding.embeddings[1]))

    def test_dtype(self):
        idx = torch.randint(
            low=0, high=100, size=(10, 20), device=self.device
        )  # some example indices

        output = self.embedding(idx)

        # Check the data type of output
        self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
