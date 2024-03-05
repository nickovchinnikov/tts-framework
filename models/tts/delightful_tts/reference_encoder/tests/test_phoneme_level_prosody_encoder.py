import unittest

import torch
from torch import nn

from models.config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)
from models.helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
)
from models.tts.delightful_tts.attention.conformer_multi_headed_self_attention import (
    ConformerMultiHeadedSelfAttention,
)
from models.tts.delightful_tts.reference_encoder import (
    PhonemeLevelProsodyEncoder,
    UtteranceLevelProsodyEncoder,
)
from models.tts.delightful_tts.reference_encoder.reference_encoder import (
    ReferenceEncoder,
)


# It checks for most of the acoustic model code
# Here you can understand the input and output shapes of the PhonemeLevelProsodyEncoder
# Integration test
class TestPhonemeLevelProsodyEncoder(unittest.TestCase):
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
            self.preprocess_config, self.model_config, n_speakers,
        )

        # Generate mock data for the forward pass
        self.forward_train_params = init_forward_trains_params(
            self.model_config,
            self.acoustic_pretraining_config,
            self.preprocess_config,
            n_speakers,
        )

        preprocess_config = self.preprocess_config
        model_config = self.model_config

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )

        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )

        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
        )

        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
        )

        self.model = PhonemeLevelProsodyEncoder(
            self.preprocess_config,
            self.model_config,
        )

    def test_model_attributes(self):
        # Test model type
        self.assertIsInstance(self.model, nn.Module)

        # Test individual components of the model
        self.assertIsInstance(self.model.encoder, ReferenceEncoder)
        self.assertIsInstance(self.model.encoder_prj, nn.Linear)
        self.assertIsInstance(self.model.attention, ConformerMultiHeadedSelfAttention)
        self.assertIsInstance(self.model.encoder_bottleneck, nn.Linear)

    def test_forward(self):
        x = torch.randn(1, 11, self.model_config.encoder.n_hidden)
        mels = torch.randn(1, self.preprocess_config.stft.n_mel_channels, 58)
        mel_lens = torch.tensor([58])
        src_mask = torch.zeros(11).bool()
        encoding = torch.randn(1, 58, self.model_config.encoder.n_hidden)

        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(
                mels=mels,
                mel_lens=mel_lens,
            ),
        )

        # Assert the shape of u_prosody_ref
        self.assertEqual(
            u_prosody_ref.shape,
            torch.Size(
                [
                    self.model_config.lang_embed_dim,
                    self.model_config.lang_embed_dim,
                    self.model_config.reference_encoder.bottleneck_size_u,
                ],
            ),
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding,
            ),
        )

        # Assert the shape of p_prosody_ref
        self.assertEqual(
            p_prosody_ref.shape,
            torch.Size(
                [
                    self.model_config.lang_embed_dim,
                    11,
                    self.model_config.reference_encoder.bottleneck_size_p,
                ],
            ),
        )


if __name__ == "__main__":
    unittest.main()
