import unittest

import torch

from models.tts.delightful_tts.conv_blocks.bsconv import BSConv1d


class TestBSConv1d(unittest.TestCase):
    def test_given_kernel_size_and_padding(self):
        # Batch size, Input channels, output channels
        N, C_in, C_out = 16, 4, 8

        for kernel_size, padding in [(5, 0), (7, 3), (11, 5)]:
            bsconv = BSConv1d(
                C_in,
                C_out,
                kernel_size,
                padding,
            )

            t_width = 100
            x = torch.randn(
                N,
                C_in,
                t_width,
            )

            out = bsconv(x)
            new_t_width = (t_width + 2 * padding - (kernel_size - 1) - 1) + 1

            self.assertEqual(
                out.shape,
                (N, C_out, new_t_width),
                f"For kernel_size={kernel_size} and padding={padding}, expected output shape: {N, C_out, new_t_width}, but got: {out.shape}",
            )

    def test_with_different_batch_size_and_input_channels(self):
        # Output channels, kernel size, padding
        C_out, kernel_size, padding = 16, 3, 1

        for N, C_in in [(32, 8), (64, 16), (128, 32)]:
            bsconv = BSConv1d(
                C_in,
                C_out,
                kernel_size,
                padding,
            )

            t_width = 100
            x = torch.randn(
                N,
                C_in,
                t_width,
            )

            out = bsconv(x)

            self.assertEqual(
                out.shape,
                (N, C_out, t_width),
                f"For batch_size={N} and input_channels={C_in}, expected output shape: {N, C_out, 100}, but got: {out.shape}",
            )


if __name__ == "__main__":
    unittest.main()
