import unittest

import torch

from model.attention.multi_head_attention import MultiHeadAttention

from helpers.tools import get_device


class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        torch.set_default_device(get_device())
        # Initialize an instance of MultiHeadAttention
        self.attention = MultiHeadAttention(
            query_dim=512, key_dim=512, num_units=512, num_heads=8
        )
        # Assuming batch=3, seq_length=10, dim=512
        self.dim_params = (3, 10, 512)
        self.query = torch.rand(self.dim_params)  # [N, T_q, query_dim]
        self.key = torch.rand(self.dim_params)  # [N, T_k, key_dim]

    def test_forward(self):
        # Test the forward function
        out = self.attention(self.query, self.key)

        # Assert output shape
        self.assertEqual(out.shape, self.dim_params)

    def test_dtype(self):
        # Test forward function
        out = self.attention(self.query, self.key)

        # Check the data type of output
        self.assertTrue(out.dtype == torch.float32)

    def test_consistent_output(self):
        # Test forward function
        out1 = self.attention(self.query, self.key)
        out2 = self.attention(self.query, self.key)

        # Check if the output is consistent for the same input
        self.assertTrue(torch.allclose(out1, out2))


# Run the tests
if __name__ == "__main__":
    unittest.main()
