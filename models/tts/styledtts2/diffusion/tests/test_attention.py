import unittest

import torch

from models.tts.styledtts2.diffusion.attention import (
    Attention,
    AttentionBase,
    FeedForward,
    RelativePositionBias,
)


class TestAttention(unittest.TestCase):
    def test_relative_position_bias(self):
        bias = RelativePositionBias(num_buckets=10, max_distance=5, num_heads=2)
        output = bias(5, 5)
        self.assertEqual(output.shape, (1, 2, 5, 5))

    def test_feed_forward(self):
        ff = FeedForward(features=10, multiplier=2)
        x = torch.randn(5, 10)
        y = ff(x)
        self.assertEqual(y.shape, (5, 10))

    def test_attention_base(self):
        attn = AttentionBase(features=10, head_features=5, num_heads=2, use_rel_pos=False)
        q = torch.randn(2, 5, 10)
        k = torch.randn(2, 5, 10)
        v = torch.randn(2, 5, 10)
        y = attn(q, k, v)
        self.assertEqual(y.shape, (2, 5, 10))

    def test_attention(self):
        attn = Attention(features=10, head_features=5, num_heads=2, use_rel_pos=False)
        x = torch.randn(2, 5, 10)
        y = attn(x)
        self.assertEqual(y.shape, (2, 5, 10))

if __name__ == "__main__":
    unittest.main()
