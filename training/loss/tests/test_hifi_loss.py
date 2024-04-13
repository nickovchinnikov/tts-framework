import unittest

import torch
from torch import Tensor

from training.loss.hifi_loss import HifiLoss


class TestHifiLoss(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.loss_module = HifiLoss()

    def test_forward(self):
        # Create some fake input data
        audio = torch.randn(1, 1, 22050)
        fake_audio = torch.randn(1, 1, 22050)
        mpd_res = (
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
        )
        msd_res = (
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
            [torch.randn(1, 1, 22050)] * 4,
        )

        # Call the forward method
        output = self.loss_module.forward(audio, fake_audio, mpd_res, msd_res)

        # Check that the output is a tuple with the expected length
        self.assertIsInstance(output, tuple)
        self.assertEqual(len(output), 9)

        # Check that each element of the output is a Tensor
        for element in output:
            self.assertIsInstance(element, Tensor)

        # Assert the value of losses
        self.assertTrue(all(element >= 0 for element in output))


if __name__ == "__main__":
    unittest.main()
