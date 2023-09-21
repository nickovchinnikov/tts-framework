import unittest
import torch


from model.attention import ConformerMultiHeadedSelfAttention


# Test class for the ConformerMultiHeadedSelfAttention class
class TestConformerMultiHeadedSelfAttention(unittest.TestCase):
    def test_forward(self):
        # Create an instance of ConformerMultiHeadedSelfAttention
        model = ConformerMultiHeadedSelfAttention(
            512, 2, 0.1
        )  # 512 dim, 2 heads, 10% dropout

        # Generate some random data for input
        batch_size = 2
        seq_length = 15
        query = torch.rand(batch_size, seq_length, 512)
        key = torch.rand(batch_size, seq_length, 512)
        value = torch.rand(batch_size, seq_length, 512)
        mask = torch.ones(batch_size, 1, seq_length)
        encoding = torch.rand(1, seq_length, 512)

        # Execute the forward pass
        outputs, attn = model(query, key, value, mask, encoding)

        # Check the output shapes
        self.assertEqual(outputs.shape, (batch_size, seq_length, 512))
        self.assertEqual(attn.shape, (batch_size, 2, seq_length, seq_length))


# Run the tests
if __name__ == "__main__":
    unittest.main()
