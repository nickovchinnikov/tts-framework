import unittest

import torch

from models.tts.styledtts2.diffusion.ada_layer_norm import AdaLayerNorm


class TestAdaLayerNorm(unittest.TestCase):
    def test_forward(self):
        # Create an instance of AdaLayerNorm
        layer = AdaLayerNorm(style_dim=5, channels=3)

        # Create a mock input tensor and style tensor
        x = torch.randn(10, 20, 3)  # batch_size=10, num_samples=20, num_channels=3
        s = torch.randn(10, 5)  # batch_size=10, style_dim=5

        # Pass the input and style tensors through the layer
        y = layer(x, s)

        # Check that the output has the correct shape
        self.assertEqual(y.shape, (10, 20, 3))

        # Check that the output is not equal to the input (since the layer should modify the input)
        self.assertFalse(torch.equal(y, x))

if __name__ == "__main__":
    unittest.main()
