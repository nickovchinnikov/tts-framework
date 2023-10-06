import unittest

import torch

from model.univnet.stft_loss import STFTLoss


class TestSTFTLoss(unittest.TestCase):
    def test_stft_loss(self):
        torch.random.manual_seed(0)
        # Test the STFT loss function with random input tensors
        loss_fn = STFTLoss()

        x = torch.randn(
            4,
            16000,
        )
        y = torch.randn(
            4,
            16000,
        )

        sc_loss, mag_loss = loss_fn(x, y)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))
        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

    def test_stft_loss_nonzero(self):
        # Test the STFT loss function with non-zero loss
        loss_fn = STFTLoss()

        # Reproducibility
        torch.manual_seed(0)

        x_mag = torch.randn(4, 16000, dtype=torch.float32)
        y_mag = torch.randn(4, 16000, dtype=torch.float32)

        sc_loss, mag_loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))

        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

        self.assertGreater(sc_loss, 0.0)
        self.assertGreater(mag_loss, 0.0)

        expected_sc = torch.tensor(0.6559)
        self.assertTrue(torch.allclose(sc_loss, expected_sc, rtol=1e-4, atol=1e-4))

        expected_mag = torch.tensor(0.6977)
        self.assertTrue(torch.allclose(mag_loss, expected_mag, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
