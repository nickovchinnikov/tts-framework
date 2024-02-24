import unittest

import torch

from training.loss import LogSTFTMagnitudeLoss


class TestLogSTFTMagnitudeLoss(unittest.TestCase):
    def test_log_stft_magnitude_loss(self):
        # Test the log STFT magnitude loss function with random input tensors
        loss_fn = LogSTFTMagnitudeLoss()

        x_mag = torch.randn(4, 100, 513)
        y_mag = torch.randn(4, 100, 513)

        loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_log_stft_magnitude_loss_nonzero(self):
        # Test the log STFT magnitude loss function with non-zero loss
        loss_fn = LogSTFTMagnitudeLoss()

        x_mag = torch.tensor([[1, 4, 9, 64], [1, 1, 1, 2]])
        y_mag = torch.tensor([[1, 8, 16, 256], [1, 1, 2, 2]])

        loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

        expected = torch.tensor(0.4185)
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
