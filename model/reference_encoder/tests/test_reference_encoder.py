import unittest
from unittest.mock import Mock

import torch

from model.reference_encoder import ReferenceEncoder
from config import PreprocessingConfig


class TestReferenceEncoder(unittest.TestCase):
    def setUp(self):

        self.preprocess_config = Mock(stft=Mock(
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=100,
            mel_fmin=20,
            mel_fmax=11025,
        ))
        self.model_config = Mock(
            reference_encoder=Mock(
                bottleneck_size_p=4,
                bottleneck_size_u=256,
                ref_enc_filters=[32, 32, 64, 64, 128, 128],
                ref_enc_size=3,
                ref_enc_strides=[1, 2, 1, 2, 1],
                ref_enc_pad=[1, 1],
                ref_enc_gru_size=32,
                ref_attention_dropout=0.2,
                token_num=32,
                predictor_kernel_size=5,
            )
        )
        self.model = ReferenceEncoder(self.preprocess_config, self.model_config)

    def test_forward_shape(self):
        # Define test case
        x = torch.randn(16, self.model.n_mel_channels, 128)  # assuming the input sequence length is 128
        mel_lens = torch.ones(16).type(torch.LongTensor) * 128  # assuming all sequences are of equal length

        # Call the forward method
        out, memory, mel_masks = self.model(x, mel_lens)
        
        # Verify the outputs
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.size(0), 16)
        self.assertEqual(out.size(2), self.model.gru.hidden_size)

        self.assertIsInstance(memory, torch.Tensor)
        self.assertEqual(memory.size(0), 1)
        self.assertEqual(memory.size(2), self.model.gru.hidden_size)

        self.assertIsInstance(mel_masks, torch.Tensor)
        self.assertEqual(list(mel_masks.size()), [16, 32])

    def test_different_mel_lens(self):
        x = torch.randn(16, self.model.n_mel_channels, 128) 
        mel_lens = torch.randint(low=1, high=129, size=(16,))

        out, memory, mel_masks = self.model(x, mel_lens)

        self.assertEqual(out.size(0), 16)
        self.assertEqual(out.size(2), self.model.gru.hidden_size)

    def test_different_batch_sizes(self):
        for batch_size in [1, 5, 10, 50]:
            x = torch.randn(batch_size, self.model.n_mel_channels, 128)
            mel_lens = torch.ones(batch_size).type(torch.LongTensor) * 128

            out, memory, mel_masks = self.model(x, mel_lens)

            self.assertEqual(out.size(0), batch_size)
            self.assertEqual(out.size(2), self.model.gru.hidden_size)


if __name__ == "__main__":
    unittest.main()
