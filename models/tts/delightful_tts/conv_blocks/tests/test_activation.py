import unittest

import torch
from torch.autograd.gradcheck import gradcheck

from models.tts.delightful_tts.conv_blocks.activation import GLUActivation


# Unit Testing Class
class TestGLUActivation(unittest.TestCase):
    def setUp(self):
        self.glu = GLUActivation()

    # Test that dimensions remain unchanged
    def test_dimensions(self):
        # random data with shape like (batch_size, channels, height, width)
        x = torch.randn(32, 4, 64, 64)
        x_after_glu = self.glu(x)
        expected_shape = (
            x_after_glu.shape[0],
            x_after_glu.shape[1] * 2,
            x_after_glu.shape[2],
            x_after_glu.shape[3],
        )
        self.assertEqual(x.shape, expected_shape)

    # Test the gradients
    def test_gradcheck(self):
        # use double precision for gradcheck
        x = torch.randn(2, 2, dtype=torch.float64, requires_grad=True)
        self.assertTrue(gradcheck(self.glu, x), "Gradient check failed")

    # Test for specific values
    def test_values(self):
        x = torch.tensor([[-0.5, -0.5], [0.5, 0.5]], dtype=torch.float32)
        x_after_glu = self.glu(x)
        expected_values = torch.tensor([[0.075], [0.25]], dtype=torch.float32)
        torch.testing.assert_close(
            x_after_glu, expected_values, rtol=1e-4, atol=1e-8,
        )


if __name__ == "__main__":
    unittest.main()
