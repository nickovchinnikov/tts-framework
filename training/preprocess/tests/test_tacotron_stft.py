import unittest

import torch

from training.preprocess.tacotron_stft import TacotronSTFT


class TestTacotronSTFT(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)

        self.batch_size = 2
        self.seq_len = 100
        self.filter_length = 1024
        self.hop_length = 256
        self.win_length = 1024
        self.n_mel_channels = self.filter_length // 2 + 1
        self.sampling_rate = 22050
        self.mel_fmin = 0
        self.mel_fmax = 8000
        self.center = True

        self.model = TacotronSTFT(
            filter_length=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mel_channels=self.n_mel_channels,
            sampling_rate=self.sampling_rate,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax,
            center=self.center,
        )

        self.device = self.model.device

    def test_spectrogram(self):
        # Test the _spectrogram method
        y = torch.randn(self.batch_size, self.filter_length // 2).to(self.device)
        y = y / torch.max(torch.abs(y))

        spec = self.model._spectrogram(y)

        self.assertEqual(
            spec.shape,
            (self.batch_size, self.filter_length // 2 + 1, 6, self.batch_size),
        )

    def test_linear_spectrogram(self):
        # Test the linear_spectrogram method
        y = torch.randn(self.batch_size, self.filter_length // 2).to(self.device)
        y = y / torch.max(torch.abs(y))

        spec = self.model.linear_spectrogram(y)

        self.assertEqual(spec.shape, (self.batch_size, self.filter_length // 2 + 1, 6))

    def test_forward(self):
        # Test the forward method
        y = torch.randn(self.n_mel_channels, self.filter_length // 2).to(self.device)
        y = y / torch.max(torch.abs(y))

        spec, mel = self.model(y)

        self.assertEqual(
            spec.shape,
            (self.filter_length // 2 + 1, self.filter_length // 2 + 1, 6),
        )
        self.assertEqual(
            mel.shape,
            (self.filter_length // 2 + 1, self.filter_length // 2 + 1, 6),
        )

    def test_spectral_normalize_torch(self):
        # Test the spectral_normalize_torch method
        magnitudes = torch.randn(self.batch_size, self.n_mel_channels, self.seq_len).to(
            self.device
        )
        output = self.model.spectral_normalize_torch(magnitudes)
        self.assertEqual(
            output.shape, (self.batch_size, self.n_mel_channels, self.seq_len)
        )

    def test_dynamic_range_compression_torch(self):
        # Test the dynamic_range_compression_torch method
        x = torch.randn(self.batch_size, self.n_mel_channels, self.seq_len).to(
            self.device
        )
        output = self.model.dynamic_range_compression_torch(x)
        self.assertEqual(
            output.shape, (self.batch_size, self.n_mel_channels, self.seq_len)
        )


if __name__ == "__main__":
    unittest.main()
