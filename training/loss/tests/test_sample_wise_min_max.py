import unittest

import torch

from training.loss.fast_speech_2_loss_gen import sample_wise_min_max


class TestSampleWiseMinMax(unittest.TestCase):
    def test_sample_wise_min_max(self):
        # Test case 1: Test with batch size 1
        x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        normalized = sample_wise_min_max(x)
        expected = torch.tensor(
            [[[0.0, 0.125, 0.25], [0.375, 0.5, 0.625], [0.75, 0.875, 1.0]]]
        )

        self.assertTrue(torch.allclose(normalized, expected))

        # Test case 2: Test with batch size 2
        x = torch.tensor(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[2, 4, 6], [8, 10, 12], [14, 16, 18]],
            ]
        )
        normalized = sample_wise_min_max(x)
        expected = torch.tensor(
            [
                [[0.0, 0.125, 0.25], [0.375, 0.5, 0.625], [0.75, 0.875, 1.0]],
                [[0.0, 0.125, 0.25], [0.375, 0.5, 0.625], [0.75, 0.875, 1.0]],
            ]
        )

        self.assertTrue(torch.allclose(normalized, expected))

        # Test case 3: Test with negative values
        x = torch.tensor(
            [
                [[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]],
                [[-2, -4, -6], [-8, -10, -12], [-14, -16, -18]],
            ]
        )
        normalized = sample_wise_min_max(x)
        expected = torch.tensor(
            [
                [[1.0, 0.875, 0.75], [0.625, 0.5, 0.375], [0.25, 0.125, 0.0]],
                [[1.0, 0.875, 0.75], [0.625, 0.5, 0.375], [0.25, 0.125, 0.0]],
            ]
        )
        self.assertTrue(torch.allclose(normalized, expected))


if __name__ == "__main__":
    unittest.main()
