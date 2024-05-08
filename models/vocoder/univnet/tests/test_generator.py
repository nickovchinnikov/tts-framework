import unittest

import torch
from torch import nn

from models.config import PreprocessingConfigUnivNet as PreprocessingConfig
from models.config import VocoderModelConfig
from models.vocoder.univnet.generator import Generator


class TestUnivNet(unittest.TestCase):
    def setUp(self):
        self.batch_size = 3
        self.in_length = 100

        self.model_config = VocoderModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        self.generator = Generator(self.model_config, self.preprocess_config)

        self.c = torch.randn(
            self.batch_size,
            self.preprocess_config.stft.n_mel_channels,
            self.in_length,
        )

    def test_forward(self):
        output = self.generator(self.c)

        # Assert the shape
        expected_shape = (self.batch_size, 1, self.in_length * 256)
        self.assertEqual(output.shape, expected_shape)

    def test_generator_inference_output_shape(self):
        mel_lens = torch.tensor([self.in_length] * self.batch_size)

        output = self.generator.infer(self.c, mel_lens)

        # Assert the shape
        expected_shape = (
            self.batch_size,
            1,
            self.in_length * self.preprocess_config.stft.hop_length,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_eval(self):
        generator = Generator(
            self.model_config,
            self.preprocess_config,
        )

        generator.eval(inference=True)
        for module in generator.modules():
            if isinstance(module, nn.Conv1d):
                self.assertFalse(hasattr(module, "weight_g"))
                self.assertFalse(hasattr(module, "weight_v"))

    def test_remove_weight_norm(self):
        generator = Generator(
            self.model_config,
            self.preprocess_config,
        )

        generator.remove_weight_norm()
        for module in generator.modules():
            if isinstance(module, nn.Conv1d):
                self.assertFalse(hasattr(module, "weight_g"))
                self.assertFalse(hasattr(module, "weight_v"))


if __name__ == "__main__":
    unittest.main()
