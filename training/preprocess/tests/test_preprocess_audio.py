import unittest

import torch

from model.config import PreprocessingConfig
from training.preprocess import PreprocessAudio


class TestPreprocessAudio(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.preprocess_config = PreprocessingConfig("english_only")
        self.preprocess_audio = PreprocessAudio(self.preprocess_config)

    def test_forward(self):
        # Set the sampling rate and duration of the audio signal
        sr_actual = 44100
        duration = 1.0

        # Set the frequency of the pitch (in Hz)
        pitch_freq = 440.0

        # Generate a time vector for the audio signal
        t = torch.linspace(0, duration, int(sr_actual * duration))

        # Generate a sinusoidal waveform with the specified pitch frequency
        audio = torch.sin(2 * torch.pi * pitch_freq * t)

        audio = audio.unsqueeze(0)

        raw_text = "Hello, world!"
        output = self.preprocess_audio(audio, sr_actual, raw_text)

        self.assertIsInstance(output, dict)

        self.assertIn("wav", output)
        self.assertIn("mel", output)
        self.assertIn("pitch", output)
        self.assertIn("phones", output)
        self.assertIn("raw_text", output)
        self.assertIn("pitch_is_normalized", output)

        self.assertIsInstance(output["wav"], torch.FloatTensor)
        self.assertIsInstance(output["mel"], torch.FloatTensor)
        self.assertIsInstance(output["pitch"], torch.Tensor)
        self.assertIsInstance(output["phones"], list)
        self.assertIsInstance(output["raw_text"], str)
        self.assertIsInstance(output["pitch_is_normalized"], bool)

        self.assertEqual(output["wav"].shape, torch.Size([22050]))
        self.assertEqual(output["mel"].shape, torch.Size([100, 86]))
        self.assertEqual(output["pitch"].shape, torch.Size([86]))
        self.assertEqual(
            output["phones"],
            [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33],
        )
        self.assertEqual(output["raw_text"], "Hello, world!")
        self.assertFalse(output["pitch_is_normalized"])

        # Load the expected output from file
        expected_output = torch.load("./mocks/preprocess_audio_output.pt")
        # Compare the loaded and output dictionaries
        torch.testing.assert_allclose(expected_output["wav"], output["wav"])
        torch.testing.assert_allclose(expected_output["mel"], output["mel"])
        torch.testing.assert_allclose(expected_output["pitch"], output["pitch"])

        self.assertEqual(expected_output["phones"], output["phones"])
        self.assertEqual(expected_output["raw_text"], output["raw_text"])
        self.assertEqual(
            expected_output["pitch_is_normalized"], output["pitch_is_normalized"]
        )

    def test_forward_with_short_audio(self):
        audio = torch.randn(1, 22050)
        sr_actual = 22050
        raw_text = "Hello, world!"
        output = self.preprocess_audio(audio, sr_actual, raw_text)

        self.assertIsNone(output)

    def test_forward_with_long_audio(self):
        audio = torch.randn(1, 88200)
        sr_actual = 44100
        raw_text = "Hello, world!"
        output = self.preprocess_audio(audio, sr_actual, raw_text)

        self.assertIsNone(output)


if __name__ == "__main__":
    unittest.main()
