import unittest

import torch
from torch import nn

from models.tts.delightful_tts.conv_blocks.bsconv import BSConv1d
from models.tts.delightful_tts.conv_blocks.conv1d import (
    DepthWiseConv1d,
    PointwiseConv1d,
)
from models.tts.delightful_tts.conv_blocks.conv1d_glu import Conv1dGLU


class TestConv1dGLU(unittest.TestCase):
    def test_initialization(self):
        """Test to check if the Conv1dGLU instance is properly created."""
        d_model, kernel_size, padding, embedding_dim = 16, 3, 1, 32
        conv_glu = Conv1dGLU(
            d_model,
            kernel_size,
            padding,
            embedding_dim,
        )

        # Checking attribute types
        self.assertIsInstance(
            conv_glu.bsconv1d,
            BSConv1d,
            msg="Expected bsconv1d attribute to be instance of BSConv1d Layer",
        )
        self.assertIsInstance(
            conv_glu.bsconv1d.pointwise,
            PointwiseConv1d,
            msg="Expected bsconv1d.pointwise to be instance of PointwiseConv1d",
        )
        self.assertIsInstance(
            conv_glu.bsconv1d.depthwise,
            DepthWiseConv1d,
            msg="Expected bsconv1d.depthwise to be instance of DepthWiseConv1d",
        )
        self.assertIsInstance(
            conv_glu.embedding_proj,
            nn.Linear,
            msg="Expected embedding_proj attribute to be instance of nn.Linear Layer",
        )

        # Checking the Conv1d, PointwiseConv1d, and DepthWiseConv1d configurations
        self.assertEqual(conv_glu.bsconv1d.pointwise.conv.in_channels, d_model)
        self.assertEqual(conv_glu.bsconv1d.pointwise.conv.out_channels, 2 * d_model)
        self.assertEqual(conv_glu.bsconv1d.depthwise.conv.in_channels, 2 * d_model)
        self.assertEqual(conv_glu.bsconv1d.depthwise.conv.out_channels, 2 * d_model)
        self.assertEqual(conv_glu.bsconv1d.depthwise.conv.kernel_size[0], kernel_size)
        self.assertEqual(conv_glu.bsconv1d.depthwise.conv.padding[0], padding)

    def test_output_shape(self):
        """Test to ensure that the output tensor shape from the forward function is as expected."""
        batch_size, sequence_length, d_model, kernel_size, padding, embedding_dim = (
            64,
            50,
            8,
            3,
            1,
            16,
        )
        conv_glu = Conv1dGLU(
            d_model,
            kernel_size,
            padding,
            embedding_dim,
        )

        # Create input tensor 'x' of shape (batch_size, sequence_length, embed_dim)
        x = torch.randn(
            batch_size,
            sequence_length,
            d_model,
        )

        # Create embeddings tensor of shape (batch_size, d_model, embedding_dim)
        embeddings = torch.randn(
            batch_size,
            sequence_length,
            embedding_dim,
        )

        out = conv_glu(x, embeddings)

        # Assuming embeddings transformations don't change the width of the input
        self.assertEqual(
            out.shape,
            (batch_size, sequence_length, d_model),
            f"Expected output shape: {batch_size, sequence_length, d_model}, but got: {out.shape}",
        )


if __name__ == "__main__":
    unittest.main()
