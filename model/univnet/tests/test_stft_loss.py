import torch
import unittest

from model.helpers.tools import get_device

from model.univnet.stft_loss import STFTLoss


class TestSTFTLoss(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_stft_loss(self):
        # Test the STFT loss function with random input tensors
        loss_fn = STFTLoss(device=self.device)

        x = torch.randn(4, 16000, device=self.device)
        y = torch.randn(4, 16000, device=self.device)

        sc_loss, mag_loss = loss_fn(x, y)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))
        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

    def test_stft_loss_nonzero(self):
        # Test the STFT loss function with non-zero loss
        loss_fn = STFTLoss(device=self.device)

        # Reproducibility
        torch.manual_seed(0)

        x_mag = torch.randn(4, 16000, device=self.device, dtype=torch.float32)
        y_mag = torch.randn(4, 16000, device=self.device, dtype=torch.float32)

        sc_loss, mag_loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))

        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

        self.assertGreater(sc_loss, 0.0)
        self.assertGreater(mag_loss, 0.0)

        expected_sc = torch.tensor(0.6628)
        self.assertTrue(torch.allclose(sc_loss, expected_sc, rtol=1e-4, atol=1e-4))

        expected_mag = torch.tensor(0.7015)
        self.assertTrue(torch.allclose(mag_loss, expected_mag, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
