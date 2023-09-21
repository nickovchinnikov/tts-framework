import unittest

import torch.nn as nn

from model.reference_encoder.reference_encoder import ReferenceEncoder
from model.attention.conformer_multi_headed_self_attention import (
    ConformerMultiHeadedSelfAttention,
)

from config import AcousticENModelConfig, PreprocessingConfig

from model.reference_encoder.phoneme_level_prosody_encoder import (
    PhonemeLevelProsodyEncoder,
)


# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
# Test class for the PhonemeLevelProsodyEncoder class
class TestPhonemeLevelProsodyEncoder(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        self.model = PhonemeLevelProsodyEncoder(
            self.preprocess_config, self.model_config
        )

    def test_model_attributes(self):
        # Test model type
        self.assertIsInstance(self.model, nn.Module)

        # Test individual components of the model
        self.assertIsInstance(self.model.encoder, ReferenceEncoder)
        self.assertIsInstance(self.model.encoder_prj, nn.Linear)
        self.assertIsInstance(self.model.attention, ConformerMultiHeadedSelfAttention)
        self.assertIsInstance(self.model.encoder_bottleneck, nn.Linear)

        # Test model's hidden dimensions
        self.assertEqual(self.model.E, self.model_config.encoder.n_hidden)
        self.assertEqual(self.model.E, self.model.d_q)
        self.assertEqual(self.model.E, self.model.d_k)


if __name__ == "__main__":
    unittest.main()
