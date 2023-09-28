import torch
import unittest

from model.univnet.spectral_convergence_loss import SpectralConvergengeLoss


class TestSpectralConvergengeLoss(unittest.TestCase):
    def test_spectral_convergence_loss(self):
        # Test the spectral convergence loss function with random input tensors
        loss_fn = SpectralConvergengeLoss()

        x_mag = torch.randn(4, 100, 513)
        y_mag = torch.randn(4, 100, 513) * 0.1

        loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertGreater(loss, 0.0)

    def test_spectral_convergence_small_vectors(self):
        # Test the spectral convergence loss function with non-zero loss
        loss_fn = SpectralConvergengeLoss()

        x_mag = torch.tensor([1, 4, 9, 64], dtype=torch.float32)
        y_mag = torch.tensor([1, 8, 16, 256], dtype=torch.float32)

        loss = loss_fn(x_mag, y_mag)

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, torch.Size([]))

        expected = torch.tensor(0.7488)
        self.assertTrue(torch.allclose(loss, expected, rtol=1e-4, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
