import unittest

import torch

from models.tts.styledtts2.diffusion.embeddings import (
    FixedEmbedding,
    LearnedPositionalEmbedding,
    SinusoidalEmbedding,
    TimePositionalEmbedding,
)


class TestEmbeddings(unittest.TestCase):
    def test_sinusoidal_embedding(self):
        emb = SinusoidalEmbedding(dim=10)
        x = torch.tensor([1.0])
        y = emb(x)
        self.assertEqual(y.shape, (1, 10))

    def test_learned_positional_embedding(self):
        emb = LearnedPositionalEmbedding(dim=10)
        x = torch.tensor([1.0])
        y = emb(x)
        self.assertEqual(y.shape, (1, 11))

    def test_time_positional_embedding(self):
        emb = TimePositionalEmbedding(dim=10, out_features=20)
        x = torch.tensor([1.0])
        y = emb(x)
        self.assertEqual(y.shape, (1, 20))

    def test_fixed_embedding(self):
        emb = FixedEmbedding(max_length=10, features=20)
        x = torch.tensor([[1, 2, 3]])
        y = emb(x)
        self.assertEqual(y.shape, (1, 3, 20))

if __name__ == "__main__":
    unittest.main()
