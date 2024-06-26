import unittest

import torch

from training.loss import MultiResolutionSTFTLoss


class TestMultiResolutionSTFTLoss(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(0)
        self.loss_fn = MultiResolutionSTFTLoss([(1024, 120, 600), (2048, 240, 1200)])

        self.x = torch.randn(
            4,
            16000,
        )
        self.y = torch.randn(
            4,
            16000,
        )

    def test_multi_resolution_stft_loss(self):
        # Test the MultiResolutionSTFTLoss class with random input tensors

        sc_loss, mag_loss = self.loss_fn(self.x, self.y)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))
        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

    def test_multi_resolution_stft_loss_nonzero(self):
        # Test the MultiResolutionSTFTLoss class with input tensors that have a non-zero loss value
        torch.manual_seed(0)

        x = torch.randn(
            4,
            16000,
        )
        y = torch.randn(
            4,
            16000,
        )

        sc_loss, mag_loss = self.loss_fn(x, y)

        self.assertIsInstance(sc_loss, torch.Tensor)
        self.assertEqual(sc_loss.shape, torch.Size([]))
        self.assertIsInstance(mag_loss, torch.Tensor)
        self.assertEqual(mag_loss.shape, torch.Size([]))

        expected_sc_loss = torch.tensor(
            0.6571,
        )
        self.assertTrue(torch.allclose(sc_loss, expected_sc_loss, atol=1e-4))

        expected_mag_loss = torch.tensor(
            0.7007,
        )
        self.assertTrue(
            torch.allclose(mag_loss, expected_mag_loss, rtol=1e-4, atol=1e-4),
        )


if __name__ == "__main__":
    unittest.main()
