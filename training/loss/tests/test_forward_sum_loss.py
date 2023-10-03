import unittest

import torch

from training.loss import ForwardSumLoss


class TestForwardSumLoss(unittest.TestCase):
    def setUp(self):
        self.forward_sum_loss = ForwardSumLoss()

    def test_forward(self):
        # Reproducible results
        torch.random.manual_seed(0)

        T = 2  # Input sequence length
        C = 2  # Number of classes (including blank)
        N = 1  # Batch size
        S = 1  # Target sequence length of longest target in batch (padding length)

        attn_logprob = torch.randn(T, N, C, C).log_softmax(2).detach().requires_grad_()

        in_lens = torch.full(size=(T,), fill_value=T, dtype=torch.long)
        out_lens = torch.randint(low=S, high=T, size=(T,), dtype=torch.long)

        loss = self.forward_sum_loss(attn_logprob, in_lens, out_lens)
        expected_loss = torch.tensor([0.0])

        self.assertTrue(torch.allclose(loss, expected_loss))


if __name__ == "__main__":
    unittest.main()
