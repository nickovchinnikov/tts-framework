import unittest
import torch

from helpers.tools import get_device

from model.attention import ConformerMultiHeadedSelfAttention


# Test class for the ConformerMultiHeadedSelfAttention class
class TestConformerMultiHeadedSelfAttention(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_forward(self):
        # Create an instance of ConformerMultiHeadedSelfAttention
        model = ConformerMultiHeadedSelfAttention(
            512, 2, 0.1, device=self.device
        )  # 512 dim, 2 heads, 10% dropout

        # Generate some random data for input
        batch_size = 2
        seq_length = 15
        query = torch.rand(batch_size, seq_length, 512, device=self.device)
        key = torch.rand(batch_size, seq_length, 512, device=self.device)
        value = torch.rand(batch_size, seq_length, 512, device=self.device)
        mask = torch.ones(
            batch_size, 1, seq_length, dtype=torch.bool, device=self.device
        )
        encoding = torch.rand(1, seq_length, 512, device=self.device)

        # Execute the forward pass
        outputs, attn = model(query, key, value, mask, encoding)

        # Assert device type
        self.assertEqual(outputs.device.type, self.device.type)
        self.assertEqual(attn.device.type, self.device.type)

        # Check the output shapes
        self.assertEqual(outputs.shape, (batch_size, seq_length, 512))
        self.assertEqual(attn.shape, (batch_size, 2, seq_length, seq_length))


# Run the tests
if __name__ == "__main__":
    unittest.main()
