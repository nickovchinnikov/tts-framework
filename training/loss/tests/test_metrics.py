import unittest

import torch

from training.loss.metrics import Metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.metrics = Metrics()

        # Set the frequency of the pitch (in Hz)
        self.pitch_freq = 440.0
        self.duration = 1.0
        self.sr = 22050

        # Generate a time vector for the audio signal
        self.t = torch.linspace(0, self.duration, int(self.sr * self.duration))

        # Generate a sinusoidal waveform with the specified pitch frequency
        self.audio = torch.sin(2 * torch.pi * self.pitch_freq * self.t).unsqueeze(0)

    def test_calculate_mcd(self):
        wav_targets = torch.randn(1, 22050)
        wav_predictions = torch.randn(1, 22050)
        mcd = self.metrics.calculate_mcd(wav_targets, wav_predictions)
        self.assertIsInstance(mcd, torch.Tensor)

    def test_calculate_spectrogram_distance(self):
        wav_targets = torch.randn(1, 22050)
        wav_predictions = torch.randn(1, 22050)
        dist = self.metrics.calculate_spectrogram_distance(wav_targets, wav_predictions)
        self.assertIsInstance(dist, torch.Tensor)

    def test_calculate_f0_rmse(self):
        wav_targets = torch.randn(1, 22050)
        wav_predictions = torch.randn(1, 22050)
        rmse = self.metrics.calculate_f0_rmse(wav_targets, wav_predictions)
        self.assertIsInstance(rmse, float)

    def test_calculate_jitter_shimmer(self):
        jitter, shimmer = self.metrics.calculate_jitter_shimmer(self.audio)
        self.assertIsInstance(jitter, float)
        self.assertIsInstance(shimmer, float)

    def test_wav_metrics(self):
        ermr, jitter, shimmer = self.metrics.wav_metrics(self.audio)
        self.assertIsInstance(ermr, float)
        self.assertIsInstance(jitter, float)
        self.assertIsInstance(shimmer, float)

if __name__ == "__main__":
    unittest.main()
