import unittest

import torch

from model.attention.feed_forward import FeedForward
from model.helpers.tools import get_device


class TestFeedForward(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        self.d_model = 10
        self.kernel_size = 3
        self.dropout = 0.2
        self.expansion_factor = 4

        self.model = FeedForward(
            self.d_model,
            self.kernel_size,
            self.dropout,
            self.expansion_factor,
            device=self.device,
        )

    def test_forward(self):
        batch_size = 5
        seq_len = 7
        num_features = self.d_model

        # Create a random tensor to act as the input
        x = torch.randn(batch_size, seq_len, num_features, device=self.device)

        # Forward pass
        output = self.model(x)

        # Assert device type
        self.assertEqual(output.device.type, self.device.type)

        # Check the resultant tensor after forward pass
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (batch_size, seq_len, num_features))


if __name__ == "__main__":
    unittest.main()
