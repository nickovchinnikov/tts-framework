import unittest

import torch

from model.config import AcousticENModelConfig, PreprocessingConfig
from model.reference_encoder import UtteranceLevelProsodyEncoder


class TestUtteranceLevelProsodyEncoder(unittest.TestCase):
    def setUp(self):
        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # Instantiate the model to be tested with the mock configurations
        self.model = UtteranceLevelProsodyEncoder(
            self.preprocess_config,
            self.model_config,
        )

    def test_forward_shape(self):
        # Define the input tensor (mels) and corresponding lengths (mel_lens)
        mels = torch.randn(
            16,
            100,
            128,
        )  # assuming the input sequence length is 128 and n_mel_channels=100
        mel_lens = (
            torch.ones(16, dtype=torch.long) * 128
        )  # assuming all sequences are of equal length

        # Make a forward pass through the model
        out = self.model(mels, mel_lens)

        # Assert the shape of the output tensor
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.shape, (16, 1, self.model.encoder_bottleneck.out_features))


if __name__ == "__main__":
    unittest.main()
