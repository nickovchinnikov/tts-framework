import unittest
import torch

from model.acoustic_model.aligner import Aligner


# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
class TestAligner(unittest.TestCase):
    def setUp(self):
        self.aligner = Aligner(d_enc_in=10, d_dec_in=10, d_hidden=20)
        self.batch_size = 5
        self.max_mel_len = 10
        self.max_text_len = 15

    def test_binarize_attention_parallel(self):
        attn = torch.randn(self.batch_size, 1, self.max_mel_len, self.max_text_len)
        in_lens = torch.randint(1, self.max_mel_len, (self.batch_size,))
        out_lens = torch.randint(1, self.max_text_len, (self.batch_size,))

        binarized_attention = self.aligner.binarize_attention_parallel(
            attn, in_lens, out_lens
        )

        self.assertIsInstance(binarized_attention, torch.Tensor)
        self.assertEqual(binarized_attention.shape, attn.shape)


if __name__ == "__main__":
    unittest.main()
