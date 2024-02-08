import unittest

import torch

from models.config import AcousticENModelConfig, PreprocessingConfig
from models.reference_encoder import ReferenceEncoder


class TestReferenceEncoder(unittest.TestCase):
    def setUp(self):
        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        self.model = ReferenceEncoder(
            self.preprocess_config,
            self.model_config,
        )

    def test_forward_shape(self):
        # Define test case
        x = torch.randn(
            16,
            self.model.n_mel_channels,
            128,
        )  # assuming the input sequence length is 128
        mel_lens = (
            torch.ones(16, dtype=torch.long) * 128
        )  # assuming all sequences are of equal length

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

    def test_different_batch_sizes(self):
        for batch_size in [1, 5, 10, 50]:
            x = torch.randn(
                batch_size,
                self.model.n_mel_channels,
                128,
            )
            mel_lens = (
                torch.ones(
                    batch_size,
                    dtype=torch.long,
                )
                * 128
            )

            out, memory, mel_masks = self.model(x, mel_lens)

            self.assertEqual(out.size(0), batch_size)
            self.assertEqual(out.size(2), self.model.gru.hidden_size)


if __name__ == "__main__":
    unittest.main()
