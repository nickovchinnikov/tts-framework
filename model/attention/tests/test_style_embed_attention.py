import unittest

import torch

from model.attention.style_embed_attention import StyleEmbedAttention
from model.helpers.tools import get_device


class TestStyleEmbedAttention(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_output_shape_and_value_range(self):
        model = StyleEmbedAttention(
            query_dim=16, key_dim=16, num_units=32, num_heads=4, device=self.device
        )
        query = torch.rand(
            5, 10, 16, device=self.device
        )  # batch of 5, 10 queries per batch, each of size 16
        key = torch.rand(
            5, 20, 16, device=self.device
        )  # batch of 5, 20 key-value pairs per batch, each of size 16
        output = model(query, key)

        # Assert device type
        self.assertEqual(output.device.type, self.device.type)

        # Check that output shape is as expected
        self.assertEqual(output.shape, (5, 10, 32))

        # Check that the values in the output tensor are within a valid range after softmax and matmul (i.e., [-1, 1]).
        self.assertTrue(torch.all((-1 < output) & (output < 1)))


# Run the tests
if __name__ == "__main__":
    unittest.main()
