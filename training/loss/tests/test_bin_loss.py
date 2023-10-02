import unittest

import torch

from training.loss import BinLoss


class TestBinLoss(unittest.TestCase):
    def setUp(self):
        self.bin_loss = BinLoss()

    def test_forward_hard_attention(self):
        # Test with hard attention
        hard_attention = torch.tensor([1, 0, 1, 0])
        soft_attention = torch.tensor([0.9, 0.1, 0.8, 0.2])

        loss = self.bin_loss(hard_attention, soft_attention)

        expected_loss = -(torch.log(torch.tensor([0.9, 0.8]))).sum() / 2

        self.assertAlmostEqual(loss.item(), expected_loss.item())

    def test_forward_soft_attention(self):
        # Test with soft attention
        hard_attention = torch.tensor([1, 0, 1, 0])
        soft_attention = torch.tensor([0.9, 0.1, 0.8, 0.2], requires_grad=True)

        loss = self.bin_loss(hard_attention, soft_attention)
        expected_loss = (
            -(torch.log(torch.tensor([0.9, 0.8], requires_grad=True))).sum() / 2
        )
        expected_loss.backward()
        loss.backward()

        self.assertAlmostEqual(loss.item(), expected_loss.item())
        self.assertTrue(
            torch.allclose(
                soft_attention.grad,
                torch.tensor([-0.5556, 0.0000, -0.6250, 0.0000]),
                atol=1e-4,
            )
        )


if __name__ == "__main__":
    unittest.main()
