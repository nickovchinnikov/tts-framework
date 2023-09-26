import unittest
import torch
import torch.nn as nn

from config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)

from helpers.initializer import (
    init_acoustic_model,
    init_conformer,
    init_forward_trains_params,
    init_mask_input_embeddings_encoding_attn_mask,
)

from helpers.tools import get_device

from model.acoustic_model.aligner import Aligner

from model.reference_encoder import (
    UtteranceLevelProsodyEncoder,
    PhonemeLevelProsodyEncoder,
)


# It checks for most of the acoustic model code
# Here you can understand the input and output shapes of the Aligner
# Integration test
class TestAligner(unittest.TestCase):
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

        self.u_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_u,
            model_config.encoder.n_hidden,
            device=self.device,
        )

        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
            device=self.device,
        )

        self.p_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_p,
            model_config.encoder.n_hidden,
            device=self.device,
        )

        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
            device=self.device,
        )

        self.aligner = Aligner(
            d_enc_in=model_config.encoder.n_hidden,
            d_dec_in=preprocess_config.stft.n_mel_channels,
            d_hidden=model_config.encoder.n_hidden,
            device=self.device,
        )

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

        # assert the device type
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
        enc_len = self.forward_train_params.enc_len

        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(
                mels=mels,
                mel_lens=mel_lens,
            )
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding
            )
        )

        x = x + self.u_bottle_out(u_prosody_ref)
        x = x + self.p_bottle_out(p_prosody_ref)

        x_res = x

        attn_logprob, attn_soft, attn_hard, attn_hard_dur = self.aligner(
            enc_in=x_res.permute((0, 2, 1)),
            dec_in=mels,
            enc_len=enc_len,
            dec_len=mel_lens,
            enc_mask=src_mask,
            attn_prior=None,
        )

        # Assert the shape of attn_logprob
        self.assertEqual(
            attn_logprob.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.model_config.lang_embed_dim,
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ]
            ),
        )

        # Assert the shape of attn_logprob==attn_soft
        self.assertEqual(
            attn_soft.shape,
            attn_logprob.shape,
        )

        # Assert the shape of attn_logprob==attn_hard
        self.assertEqual(
            attn_soft.shape,
            attn_hard.shape,
        )

        # Assert the shape of attn_logprob==attn_hard
        self.assertEqual(
            attn_hard_dur.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ]
            ),
        )

    def test_binarize_attention_parallel(self):
        aligner = Aligner(d_enc_in=10, d_dec_in=10, d_hidden=20, device=self.device)
        batch_size = 5
        max_mel_len = 10
        max_text_len = 15

        attn = torch.randn(batch_size, 1, max_mel_len, max_text_len, device=self.device)
        in_lens = torch.randint(1, max_mel_len, (batch_size,), device=self.device)
        out_lens = torch.randint(1, max_text_len, (batch_size,), device=self.device)

        binarized_attention = aligner.binarize_attention_parallel(
            attn, in_lens, out_lens
        )

        # Assert the device type
        self.assertEqual(binarized_attention.device.type, self.device.type)

        self.assertIsInstance(binarized_attention, torch.Tensor)

        # Assert the shape of binarized_attention
        self.assertEqual(
            binarized_attention.shape,
            torch.Size(
                [
                    batch_size,
                    1,
                    max_mel_len,
                    max_text_len,
                ]
            ),
        )
        self.assertEqual(binarized_attention.shape, attn.shape)


if __name__ == "__main__":
    unittest.main()
