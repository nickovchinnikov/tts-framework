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
)
import helpers.tools as tools

from model.acoustic_model.helpers import positional_encoding
from model.acoustic_model.aligner import Aligner
from model.acoustic_model.phoneme_prosody_predictor import PhonemeProsodyPredictor

from model.reference_encoder import (
    UtteranceLevelProsodyEncoder,
    PhonemeLevelProsodyEncoder,
)


# @inprogress!
# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
class TestAligner(unittest.TestCase):
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

        preprocess_config = self.preprocess_config
        model_config = self.model_config

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )
        self.utterance_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config, phoneme_level=False
        )
        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )
        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config, phoneme_level=True
        )
        self.u_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_u,
            model_config.encoder.n_hidden,
        )
        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
        )
        self.p_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_p,
            model_config.encoder.n_hidden,
        )
        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
        )

        self.aligner = Aligner(
            d_enc_in=model_config.encoder.n_hidden,
            d_dec_in=preprocess_config.stft.n_mel_channels,
            d_hidden=model_config.encoder.n_hidden,
        )

    def test_forward(self):
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

        # u_prosody_ref = self.u_norm(
        #     self.utterance_prosody_encoder(
        #         mels=mels, mel_lens=self.forward_train_params.mel_lens
        #     )
        # )

        # u_prosody_pred = self.u_norm(
        #     self.average_utterance_prosody(
        #         u_prosody_pred=self.utterance_prosody_predictor(x=x, mask=src_mask),
        #         src_mask=src_mask,
        #     )
        # )

        # p_prosody_ref = self.p_norm(
        #     self.phoneme_prosody_encoder(
        #         x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding
        #     )
        # )
        # p_prosody_pred = self.p_norm(
        #     self.phoneme_prosody_predictor(
        #         x=x,
        #         mask=src_mask,
        #     )
        # )
        # if use_ground_truth:
        #     x = x + self.u_bottle_out(u_prosody_ref)
        #     x = x + self.p_bottle_out(p_prosody_ref)
        # else:
        #     x = x + self.u_bottle_out(u_prosody_pred)
        #     x = x + self.p_bottle_out(p_prosody_pred)
        # x_res = x
        # attn_logprob, attn_soft, attn_hard, attn_hard_dur = self.aligner(
        #     enc_in=x_res.permute((0, 2, 1)),
        #     dec_in=mels,
        #     enc_len=src_lens,
        #     dec_len=mel_lens,
        #     enc_mask=src_mask,
        #     attn_prior=attn_priors,
        # )

    def test_binarize_attention_parallel(self):
        aligner = Aligner(d_enc_in=10, d_dec_in=10, d_hidden=20)
        batch_size = 5
        max_mel_len = 10
        max_text_len = 15

        attn = torch.randn(batch_size, 1, max_mel_len, max_text_len)
        in_lens = torch.randint(1, max_mel_len, (batch_size,))
        out_lens = torch.randint(1, max_text_len, (batch_size,))

        binarized_attention = aligner.binarize_attention_parallel(
            attn, in_lens, out_lens
        )

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
