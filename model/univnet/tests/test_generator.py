import unittest
import torch
import torch.nn as nn

from config import VocoderModelConfig, PreprocessingConfig

from helpers.tools import get_device

from model.univnet.generator import Generator


class TestGenerator(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.batch_size = 3
        self.in_length = 100

        self.model_config = VocoderModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        self.generator = Generator(
            self.model_config, self.preprocess_config, device=self.device
        )

        self.c = torch.randn(
            self.batch_size,
            self.preprocess_config.stft.n_mel_channels,
            self.in_length,
            device=self.device,
        )

    def test_forward(self):
        output = self.generator(self.c)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        # Assert the shape
        expected_shape = (self.batch_size, 1, self.in_length * 256)
        self.assertEqual(output.shape, expected_shape)

    def test_generator_inference_output_shape(self):
        mel_lens = torch.tensor([self.in_length] * self.batch_size).to(self.device)

        output = self.generator.infer(self.c, mel_lens)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        # Assert the shape
        expected_shape = (
            self.batch_size,
            1,
            self.in_length * self.preprocess_config.stft.hop_length,
        )
        self.assertEqual(output.shape, expected_shape)

    def test_eval(self):
        generator = Generator(
            self.model_config, self.preprocess_config, device=self.device
        )

        generator.eval(inference=True)
        for module in generator.modules():
            if isinstance(module, nn.Conv1d):
                self.assertFalse(hasattr(module, "weight_g"))
                self.assertFalse(hasattr(module, "weight_v"))

    def test_remove_weight_norm(self):
        generator = Generator(
            self.model_config, self.preprocess_config, device=self.device
        )

        generator.remove_weight_norm()
        for module in generator.modules():
            if isinstance(module, nn.Conv1d):
                self.assertFalse(hasattr(module, "weight_g"))
                self.assertFalse(hasattr(module, "weight_v"))


if __name__ == "__main__":
    unittest.main()
