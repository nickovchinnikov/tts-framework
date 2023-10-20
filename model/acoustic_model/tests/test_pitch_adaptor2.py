import unittest

import torch

from model.acoustic_model.pitch_adaptor2 import PitchAdaptor
from model.config import AcousticENModelConfig


class TestPitchAdaptor(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        # Initialize a PitchAdaptor instance
        self.model_config = AcousticENModelConfig()

        # Replace with your actual config
        self.pitch_adaptor = PitchAdaptor(self.model_config)

    def test_get_pitch_bins(self):
        pitch_bins = self.pitch_adaptor.get_pitch_bins((0, 1))
        self.assertEqual(pitch_bins.shape[0], self.pitch_adaptor.n_bins - 1)

    def test_get_pitch_embedding_train(self):
        dim = self.model_config.speaker_embed_dim

        x = torch.randn(dim, dim, dim)
        target = torch.randn(dim, dim)

        mask = torch.ones(dim, dim, dtype=torch.bool)
        (
            prediction,
            embedding_true,
            embedding_pred,
        ) = self.pitch_adaptor.get_pitch_embedding_train(x, (0, 1), target, mask)

        self.assertEqual(prediction.shape, target.shape)
        self.assertEqual(embedding_true.shape, x.shape)
        self.assertEqual(embedding_pred.shape, x.shape)

    def test_get_pitch_embedding(self):
        dim = self.model_config.speaker_embed_dim

        x = torch.randn(dim, dim, dim)
        mask = torch.ones(dim, dim, dtype=torch.bool)

        embedding = self.pitch_adaptor.get_pitch_embedding(x, (0, 1), mask, 1)
        self.assertEqual(embedding.shape, x.shape)

    def test_add_pitch_train(self):
        dim = self.model_config.speaker_embed_dim

        x = torch.randn(dim, dim, dim)
        pitch_target = torch.randn(dim, dim)

        src_mask = torch.ones(dim, dim, dtype=torch.bool)

        (
            x,
            pitch_prediction,
            pitch_embedding_true,
            pitch_embedding_pred,
        ) = self.pitch_adaptor.add_pitch_train(x, (0, 1), pitch_target, src_mask, True)

        self.assertEqual(pitch_prediction.shape, pitch_target.shape)
        self.assertEqual(pitch_embedding_true.shape, x.shape)
        self.assertEqual(pitch_embedding_pred.shape, x.shape)

    def test_add_pitch(self):
        dim = self.model_config.speaker_embed_dim

        x = torch.randn(dim, dim, dim)
        src_mask = torch.ones(dim, dim, dtype=torch.bool)

        x = self.pitch_adaptor.add_pitch(x, (0, 1), src_mask, 1)

        self.assertEqual(x.shape, (dim, dim, dim))


if __name__ == "__main__":
    unittest.main()
