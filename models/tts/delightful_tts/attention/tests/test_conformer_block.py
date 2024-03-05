import unittest

import torch

from models.config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)
from models.helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
    init_mask_input_embeddings_encoding_attn_mask,
)
from models.tts.delightful_tts.attention.conformer_block import ConformerBlock


# ConformerBlock is used in the Conformer, crucial for the training
# Here you can understand the input and output shapes of the Conformer
# Integration test
class TestConformerBlock(unittest.TestCase):
    def setUp(self):
        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        # Based on speaker.json mock
        n_speakers = 10

        # Init conformer config
        _, self.conformer_config = init_conformer(
            self.model_config,
        )

        # Add AcousticModel instance
        self.acoustic_model, _ = init_acoustic_model(
            self.preprocess_config,
            self.model_config,
            n_speakers,
        )

        self.model = ConformerBlock(
            d_model=self.conformer_config.dim,
            n_head=self.conformer_config.n_heads,
            kernel_size_conv_mod=self.conformer_config.kernel_size_conv_mod,
            embedding_dim=self.conformer_config.embedding_dim,
            dropout=self.conformer_config.p_dropout,
            with_ff=self.conformer_config.with_ff,
        )

        # Generate mock data for the forward pass
        self.forward_train_params = init_forward_trains_params(
            self.model_config,
            self.acoustic_pretraining_config,
            self.preprocess_config,
            n_speakers,
        )

    def test_initialization(self):
        """Test for successful creation of ConformerBlock instance."""
        self.assertIsInstance(self.model, ConformerBlock)

    def test_forward(self):
        """Test for successful forward pass."""
        (
            src_mask,
            x,
            embeddings,
            encoding,
            attn_mask,
        ) = init_mask_input_embeddings_encoding_attn_mask(
            self.acoustic_model,
            self.forward_train_params,
            self.model_config,
        )

        # Run conformer block model
        # x: Tensor containing the encoded sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        x = self.model(
            x,
            mask=src_mask,
            slf_attn_mask=attn_mask,
            embeddings=embeddings,
            encoding=encoding,
        )

        # Assert the shape of x
        self.assertEqual(
            x.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim // 2,
                ],
            ),
        )

    def test_with_ff_flag(self):
        """Test for correct response based on `with_ff` flag during initialization."""
        model = ConformerBlock(
            d_model=20,
            n_head=5,
            kernel_size_conv_mod=5,
            embedding_dim=20,
            dropout=0.4,
            with_ff=False,
        )
        self.assertFalse(hasattr(model, "ff"))

        model = ConformerBlock(
            d_model=12,
            n_head=6,
            kernel_size_conv_mod=3,
            embedding_dim=12,
            dropout=0.1,
            with_ff=True,
        )
        self.assertTrue(hasattr(model, "ff"))


if __name__ == "__main__":
    unittest.main()
