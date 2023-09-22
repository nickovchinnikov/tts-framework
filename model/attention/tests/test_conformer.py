import unittest

from dataclasses import dataclass

import torch

from config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)

from model.attention.conformer import Conformer

import helpers.tools as tools
from helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
)

from model.acoustic_model.helpers import positional_encoding


# It's one of the most important component test
# Conformer is used in the encoder of the AccousticModel, crucial for the training
# Here you can understand the input and output shapes of the Conformer
class TestConformer(unittest.TestCase):
    def setUp(self):
        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        # Based on speaker.json mock
        n_speakers = 10

        # # Add Conformer as encoder
        self.encoder, _ = init_conformer(self.model_config)

        # Add AcousticModel instance
        self.acoustic_model, _ = init_acoustic_model(
            self.preprocess_config, self.model_config, n_speakers
        )

        # Generate mock data for the forward pass
        self.forward_train_params = init_forward_trains_params(
            self.model_config, self.acoustic_pretraining_config, n_speakers
        )

    def test_initialization(self):
        """
        Test that a Conformer instance is correctly initialized.
        """
        self.assertIsInstance(self.encoder, Conformer)

    def test_forward(self):
        """
        Test that a Conformer instance can correctly perform a forward pass.
        For this test case we use the code from AccousticModel.
        """
        # Generate masks for padding positions in the source sequences and mel sequences
        # src_mask: Tensor containing the masks for padding positions in the source sequences. Shape: [1, batch_size]
        src_mask = tools.get_mask_from_lengths(self.forward_train_params.src_lens)

        # x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        # embeddings: Tensor containing the embeddings. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim + lang_embed_dim]
        x, embeddings = self.acoustic_model.get_embeddings(
            token_idx=self.forward_train_params.x,
            speaker_idx=self.forward_train_params.speakers,
            src_mask=src_mask,
            lang_idx=self.forward_train_params.langs,
        )

        # Assert the shape of x
        self.assertEqual(
            x.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )

        # Assert the shape of embeddings
        self.assertEqual(
            embeddings.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim
                    + self.model_config.lang_embed_dim,
                ]
            ),
        )

        # encoding: Tensor containing the positional encoding.
        # Shape: [lang_embed_dim, max(forward_train_params.mel_lens), encoder.n_hidden]
        encoding = positional_encoding(
            self.model_config.encoder.n_hidden,
            max(x.shape[1], max(self.forward_train_params.mel_lens)),
            device=x.device,
        )

        # Run conformer encoder
        # x: Tensor containing the encoded sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        # Assert the shape of x
        self.assertEqual(
            x.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
