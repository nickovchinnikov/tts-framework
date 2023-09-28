import unittest
import torch

from config import VocoderModelConfig, PreprocessingConfig

from helpers.tools import get_mask_from_lengths, get_device

from model.univnet import Generator, TracedGenerator


class TestTracedGenerator(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.batch_size = 3
        self.in_length = 100
        self.mel_channels = 80

        self.model_config = VocoderModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        self.generator = Generator(
            self.model_config, self.preprocess_config, device=self.device
        )

        self.example_inputs = (
            torch.randn(
                self.batch_size,
                self.preprocess_config.stft.n_mel_channels,
                self.in_length,
                device=self.device,
            ),
        )

        self.traced_generator = TracedGenerator(
            self.generator, self.example_inputs, device=self.device
        )

        self.c = torch.randn(
            self.batch_size,
            self.preprocess_config.stft.n_mel_channels,
            self.in_length,
            device=self.device,
        )

        self.mel_lens = torch.tensor(
            [self.in_length] * self.batch_size, device=self.device
        )

    def test_forward(self):
        output = self.traced_generator(self.c, self.mel_lens)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        # Assert the shape
        expected_shape = (self.batch_size, 1, self.in_length * 256)
        self.assertEqual(output.shape, expected_shape)

    def test_forward_with_masked_c(self):
        mel_lens = torch.tensor([self.in_length] * self.batch_size, device=self.device)

        # Mask the input mel-spectrogram tensor
        mel_mask = get_mask_from_lengths(mel_lens).unsqueeze(1)
        c = self.c.masked_fill(mel_mask, self.traced_generator.mel_mask_value)

        output = self.traced_generator(c, mel_lens)

        # Assert the device
        self.assertEqual(output.device.type, self.device.type)

        # Assert the shape
        expected_shape = (self.batch_size, 1, self.in_length * 256)
        self.assertEqual(output.shape, expected_shape)
