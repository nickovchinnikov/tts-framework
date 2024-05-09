import unittest

import torch

from models.config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
)
from models.config import (
    PreprocessingConfigUnivNet as PreprocessingConfig,
)
from models.tts.delightful_tts.acoustic_model.aligner import Aligner


# It checks for most of the acoustic model code
# Here you can understand the input and output shapes of the Aligner
# Integration test
class TestAligner(unittest.TestCase):
    def setUp(self):
        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        self.aligner = Aligner(
            d_enc_in=self.model_config.encoder.n_hidden,
            d_dec_in=self.preprocess_config.stft.n_mel_channels,
            d_hidden=self.model_config.encoder.n_hidden,
        )

    def test_forward(self):
        x_res = torch.randn(1, 11, self.model_config.encoder.n_hidden)
        mels = torch.randn(1, self.preprocess_config.stft.n_mel_channels, 58)
        src_lens = torch.tensor([11])
        mel_lens = torch.tensor([58])
        src_mask = torch.zeros(11).bool()
        attn_prior = torch.randn(1, 11, 58)

        attn_logprob, attn_soft, attn_hard, attn_hard_dur = self.aligner(
            enc_in=x_res.permute((0, 2, 1)),
            dec_in=mels,
            enc_len=src_lens,
            dec_len=mel_lens,
            enc_mask=src_mask,
            attn_prior=attn_prior,
        )

        self.assertIsInstance(attn_logprob, torch.Tensor)
        self.assertIsInstance(attn_soft, torch.Tensor)
        self.assertIsInstance(attn_hard, torch.Tensor)
        self.assertIsInstance(attn_hard_dur, torch.Tensor)

    def test_binarize_attention_parallel(self):
        aligner = Aligner(
            d_enc_in=10,
            d_dec_in=10,
            d_hidden=20,
        )
        batch_size = 5
        max_mel_len = 10
        max_text_len = 15

        attn = torch.randn(
            batch_size,
            1,
            max_mel_len,
            max_text_len,
        )
        in_lens = torch.randint(
            1,
            max_mel_len,
            (batch_size,),
        )
        out_lens = torch.randint(
            1,
            max_text_len,
            (batch_size,),
        )

        binarized_attention = aligner.binarize_attention_parallel(
            attn,
            in_lens,
            out_lens,
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
                ],
            ),
        )
        self.assertEqual(binarized_attention.shape, attn.shape)


if __name__ == "__main__":
    unittest.main()
