import unittest

import torch

from model.attention.conformer import Conformer
from model.config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)
from model.helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
    init_mask_input_embeddings_encoding_attn_mask,
)


# Conformer is used in the encoder of the AccousticModel, crucial for the training
# Here you can understand the input and output shapes of the Conformer
# Integration test
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
            self.model_config,
            self.acoustic_pretraining_config,
            self.preprocess_config,
            n_speakers,
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
        (
            src_mask,
            x,
            embeddings,
            encoding,
            _,
        ) = init_mask_input_embeddings_encoding_attn_mask(
            self.acoustic_model,
            self.forward_train_params,
            self.model_config,
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
