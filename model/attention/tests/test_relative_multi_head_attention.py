import unittest
import torch

from model.attention.relative_multi_head_attention import RelativeMultiHeadAttention

from helpers.tools import get_device


class TestRelativeMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        # Initialize an instance of RelativeMultiHeadAttention
        self.attention = RelativeMultiHeadAttention(
            d_model=512, num_heads=8, device=self.device
        )

        # Generate random tensors for query, key, value, pos_embedding, mask
        # Assuming batch=3, seq_length=10, dim=512
        self.params_shape = (3, 10, 512)
        self.query = torch.rand(self.params_shape, device=self.device)
        self.key = torch.rand(self.params_shape, device=self.device)
        self.value = torch.rand(self.params_shape, device=self.device)
        self.pos_embedding = torch.rand(self.params_shape, device=self.device)

        # A simple test case without actual masked positions
        self.mask_shape = (3, 8, 10, 10)
        self.mask = torch.zeros(self.mask_shape, device=self.device).type(torch.bool)

    def test_init_assert(self):
        # Test initializing with an invalid d_model and num_head pair
        with self.assertRaises(AssertionError):
            RelativeMultiHeadAttention(d_model=512, num_heads=7)

    def test_forward(self):
        # Generate random tensors for query, key, value, pos_embedding, mask
        # Test the forward function
        context, attn = self.attention(
            self.query, self.key, self.value, self.pos_embedding, self.mask
        )

        # Assert device type
        self.assertEqual(context.device.type, self.device.type)
        self.assertEqual(attn.device.type, self.device.type)

        # Assert output shapes
        self.assertEqual(context.shape, self.params_shape)
        self.assertEqual(attn.shape, self.mask_shape)
        # Check data types of outputs
        self.assertTrue(context.dtype == torch.float32)
        self.assertTrue(attn.dtype == torch.float32)

    def test_relative_shift(self):
        # Generate a random positional score tensor
        # Assuming batch=3, num_heads=8, seq_length1=10, seq_length2=10
        params_shape = (*self.params_shape, 10)
        pos_score = torch.rand(params_shape, device=self.device)

        # Test the _relative_shift function
        shifted_pos_score = self.attention._relative_shift(pos_score)

        # Assert output shape and values
        self.assertEqual(shifted_pos_score.shape, params_shape)
        self.assertTrue(torch.all((shifted_pos_score >= 0) & (shifted_pos_score <= 1)))

    def test_attention_values(self):
        # Test the forward function
        context, attn = self.attention(
            self.query, self.key, self.value, self.pos_embedding, self.mask
        )

        # Check values of attention output are in range [0, 1]
        self.assertTrue(torch.all((attn >= 0) & (attn <= 1)))


if __name__ == "__main__":
    unittest.main()
