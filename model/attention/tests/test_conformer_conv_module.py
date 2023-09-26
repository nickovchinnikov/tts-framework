import unittest
import torch


from model.attention.conformer_conv_module import ConformerConvModule

from helpers.tools import get_device


class TestConformerConvModule(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

    def test_forward_output_shape(self):
        """
        Test that the output shape from the forward method matches the expected shape.
        """
        d_model = 10
        kernel_size = 3
        dropout = 0.2

        model = ConformerConvModule(
            d_model, kernel_size=kernel_size, dropout=dropout, device=self.device
        )

        batch_size = 5
        seq_len = 7
        num_features = d_model

        # Create a random tensor to act as the input
        x = torch.randn(batch_size, seq_len, num_features, device=self.device)

        # Forward pass through the model
        output = model(x)

        # Assert device type
        self.assertEqual(output.device.type, self.device.type)

        # Check the output has the expected shape (batch_size, seq_len, num_features)
        self.assertEqual(output.shape, (batch_size, seq_len, num_features))


if __name__ == "__main__":
    unittest.main()
