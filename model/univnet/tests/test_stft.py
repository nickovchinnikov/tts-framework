import unittest

import torch

from model.univnet.stft import stft


class TestSTFT(unittest.TestCase):
    def test_stft(self):
        # Test the STFT function with a random input signal
        x = torch.randn(4, 16384)
        fft_size = 1024
        hop_size = 256
        win_length = 1024
        window = torch.hann_window(win_length)
        output = stft(x, fft_size, hop_size, win_length, window)
        self.assertEqual(output.shape[0], 4)
        self.assertEqual(output.shape[2], fft_size // 2 + 1)
        self.assertEqual(
            output.shape[1], (16384 - win_length) // hop_size + x.shape[0] + 1,
        )


if __name__ == "__main__":
    unittest.main()
