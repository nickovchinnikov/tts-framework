import unittest
from unittest.mock import Mock

import torch

from model.reference_encoder import UtteranceLevelProsodyEncoder


class TestUtteranceLevelProsodyEncoder(unittest.TestCase):

    @classmethod
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
            ),
            encoder=Mock(
                n_layers=4,
                n_heads=6,
                n_hidden=384,
                p_dropout=0.1,
                kernel_size_conv_mod=7,
                kernel_size_depthwise=7,
                with_ff=False,
            )
        )
        
        # Instantiate the model to be tested with the mock configurations
        self.model = UtteranceLevelProsodyEncoder(self.preprocess_config, self.model_config)

    def test_forward_shape(self):
        # Define the input tensor (mels) and corresponding lengths (mel_lens)
        mels = torch.randn(16, 100, 128)  # assuming the input sequence length is 128 and n_mel_channels=100
        mel_lens = torch.ones(16).type(torch.LongTensor) * 128  # assuming all sequences are of equal length

        # Make a forward pass through the model
        out = self.model(mels, mel_lens)

        # Assert the shape of the output tensor
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (16, 1, self.model.encoder_bottleneck.out_features))


if __name__ == "__main__":
    unittest.main()
