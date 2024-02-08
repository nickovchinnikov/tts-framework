import unittest
from unittest.mock import Mock

import torch

from models.acoustic_model import PhonemeProsodyPredictor


class TestPhonemeProsodyPredictor(unittest.TestCase):
    def setUp(self):
        model_config = Mock(
            encoder=Mock(
                n_layers=4,
                n_heads=6,
                n_hidden=384,
                p_dropout=0.1,
                kernel_size_conv_mod=7,
                kernel_size_depthwise=7,
                with_ff=False,
            ),
            reference_encoder=Mock(
                predictor_kernel_size=3,
                p_dropout=0.1,
                bottleneck_size_p=128,
                bottleneck_size_u=256,
            ),
        )
        self.model = PhonemeProsodyPredictor(model_config, phoneme_level=True)

    def test_forward(self):
        x = torch.rand(
            16,
            384,
            384,
        )
        mask = torch.zeros(16, 384).bool()

        # Calling model's forward method
        out = self.model(x, mask)

        # Check if the output is of expected type and shape
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.size(), (16, 384, 128))


if __name__ == "__main__":
    unittest.main()
