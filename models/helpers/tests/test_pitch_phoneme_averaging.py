import unittest

import torch

from models.helpers import (
    pitch_phoneme_averaging,
)


class TestPitchPhonemeAveraging(unittest.TestCase):
    def test_pitch_phoneme_averaging(self):
        # Initialize inputs
        durations = torch.tensor([[5, 1, 3, 0], [2, 4, 0, 0]], dtype=torch.float32)
        num_phonemes = durations.shape[-1]
        max_length = int(torch.sum(durations, dim=1).int().max().item())
        pitches = torch.rand(2, max_length)

        max_phoneme_len = num_phonemes

        # Call the pitch_phoneme_averaging method
        result = pitch_phoneme_averaging(durations, pitches, max_phoneme_len)

        # Assert output type
        self.assertIsInstance(result, torch.Tensor)

        # Assert output shape
        expected_shape = durations.shape
        self.assertEqual(result.shape, expected_shape)

        # Assert all pitch values are within [0,1] since input pitch values were from a uniform distribution over [0,1]
        self.assertTrue(torch.all((result >= 0) & (result <= 1)))


# Run tests
if __name__ == "__main__":
    unittest.main()
