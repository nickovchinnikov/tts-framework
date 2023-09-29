import unittest

import torch
import torch.nn as nn

from model.reference_encoder.reference_encoder import ReferenceEncoder
from model.attention.conformer_multi_headed_self_attention import (
    ConformerMultiHeadedSelfAttention,
)

from model.config import (
    AcousticENModelConfig,
    PreprocessingConfig,
    AcousticPretrainingConfig,
)

from model.reference_encoder import (
    PhonemeLevelProsodyEncoder,
    UtteranceLevelProsodyEncoder,
)

from helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
    init_mask_input_embeddings_encoding_attn_mask,
)
from helpers.tools import get_device


# It checks for most of the acoustic model code
# Here you can understand the input and output shapes of the PhonemeLevelProsodyEncoder
# Integration test
class TestPhonemeLevelProsodyEncoder(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        # Based on speaker.json mock
        n_speakers = 10

        # # Add Conformer as encoder
        self.encoder, _ = init_conformer(self.model_config, device=self.device)

        # Add AcousticModel instance
        self.acoustic_model, _ = init_acoustic_model(
            self.preprocess_config, self.model_config, n_speakers, device=self.device
        )

        # Generate mock data for the forward pass
        self.forward_train_params = init_forward_trains_params(
            self.model_config,
            self.acoustic_pretraining_config,
            self.preprocess_config,
            n_speakers,
            device=self.device,
        )

        preprocess_config = self.preprocess_config
        model_config = self.model_config

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            preprocess_config, model_config, device=self.device
        )

        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config, model_config, device=self.device
        )

        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
            device=self.device,
        )

        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
            device=self.device,
        )

        self.model = PhonemeLevelProsodyEncoder(
            self.preprocess_config, self.model_config, device=self.device
        )

    def test_model_attributes(self):
        # Test model type
        self.assertIsInstance(self.model, nn.Module)

        # Test individual components of the model
        self.assertIsInstance(self.model.encoder, ReferenceEncoder)
        self.assertIsInstance(self.model.encoder_prj, nn.Linear)
        self.assertIsInstance(self.model.attention, ConformerMultiHeadedSelfAttention)
        self.assertIsInstance(self.model.encoder_bottleneck, nn.Linear)

        # Test model's hidden dimensions
        self.assertEqual(self.model.E, self.model_config.encoder.n_hidden)
        self.assertEqual(self.model.E, self.model.d_q)
        self.assertEqual(self.model.E, self.model.d_k)

    def test_forward(self):
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

        # Run conformer encoder
        # x: Tensor containing the encoded sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        # Assert the device type of x
        self.assertEqual(x.device.type, self.device.type)

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

        # params for the testing
        mels = self.forward_train_params.mels
        mel_lens = self.forward_train_params.mel_lens

        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(
                mels=mels,
                mel_lens=mel_lens,
            )
        )

        # Assert the shape of u_prosody_ref
        self.assertEqual(
            u_prosody_ref.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.model_config.lang_embed_dim,
                    self.model_config.reference_encoder.bottleneck_size_u,
                ]
            ),
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding
            )
        )

        # Assert the shape of p_prosody_ref
        self.assertEqual(
            p_prosody_ref.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.reference_encoder.bottleneck_size_p,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
