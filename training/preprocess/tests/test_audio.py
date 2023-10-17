import math
import unittest

import numpy as np
import torch

from training.preprocess.audio import (
    normalize_loudness,
    preprocess_audio,
    resample,
    stereo_to_mono,
)


# Create a class to test the ComputePitch class
class TestAudio(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_stereo_to_mono(self):
        # Test the stereo_to_mono function with a simple example
        audio = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        expected_output = torch.tensor([[2.5, 3.5, 4.5]])
        actual_output = stereo_to_mono(audio)

        self.assertTrue(torch.allclose(actual_output, expected_output))

        # Test the stereo_to_mono function with a larger example
        audio = torch.randn(2, 44100)
        expected_output = torch.mean(audio, axis=0, keepdims=True)
        actual_output = stereo_to_mono(audio)
        self.assertTrue(torch.allclose(actual_output, expected_output))

        # Test the stereo_to_mono function with a zero-dimensional tensor
        audio = torch.tensor(1.0)
        expected_output = torch.tensor([[1.0]])
        actual_output = stereo_to_mono(audio)
        self.assertTrue(torch.allclose(actual_output, expected_output))

        # Test the stereo_to_mono function with a one-dimensional tensor
        audio = torch.tensor([1.0, 2.0, 3.0])
        expected_output = torch.tensor([2.0])
        actual_output = stereo_to_mono(audio)
        self.assertTrue(torch.allclose(actual_output, expected_output))

        # Test the stereo_to_mono function with a three-dimensional tensor
        audio = torch.randn(2, 3, 44100)
        expected_output = torch.mean(audio, axis=0, keepdims=True)
        actual_output = stereo_to_mono(audio)
        self.assertTrue(torch.allclose(actual_output, expected_output))

    def test_resample(self):
        # Test the resample function with a simple example
        wav = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        orig_sr = 44100
        target_sr = 22050

        # Test the output shape
        expected_shape = math.ceil(
            len(wav) / 2,
        )
        actual_output = resample(wav, orig_sr, target_sr)
        self.assertEqual(actual_output.shape, (expected_shape,))

    def test_preprocess_audio(self):
        # Test the preprocess_audio function with a simple example
        audio = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        sr_actual = 44100
        sr = 22050

        # Test the output shape
        expected_shape = torch.Size([2])
        actual_output, actual_sr = preprocess_audio(audio, sr_actual, sr)
        self.assertEqual(actual_output.shape, expected_shape)

        # Test the output values
        self.assertEqual(actual_sr, sr)

        # Test the preprocess_audio function with a three-dimensional tensor
        audio = torch.randn(2, 3, 44100)
        sr_actual = 44100
        sr = 22050

        # Test the output shape
        expected_shape = torch.Size([3, len(audio[0][0]) // 2])
        actual_output, actual_sr = preprocess_audio(audio, sr_actual, sr)

        self.assertEqual(actual_output.shape, expected_shape)
        self.assertEqual(actual_sr, sr)

    def test_normalize_loudness(self):
        # Test the normalize_loudness function with a simple example
        wav = torch.tensor([1.0, 2.0, 3.0])
        expected_output = torch.tensor([0.33333333, 0.66666667, 1.0])
        actual_output = normalize_loudness(wav)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

        # Test the normalize_loudness function with a larger example
        wav = torch.randn(44100)
        expected_output = wav / torch.max(torch.abs(wav))
        actual_output = normalize_loudness(wav)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

        # Test the normalize_loudness function with a zero-dimensional array
        wav = torch.tensor(1.0)
        expected_output = torch.tensor(1.0)
        actual_output = normalize_loudness(wav)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

        # Test the normalize_loudness function with a one-dimensional array
        wav = torch.tensor([1.0, 2.0, 3.0])
        expected_output = torch.tensor([0.33333333, 0.66666667, 1.0])
        actual_output = normalize_loudness(wav)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

        # Test the normalize_loudness function with a three-dimensional array
        wav = torch.randn(2, 3, 44100)
        expected_output = wav / torch.max(torch.abs(wav))
        actual_output = normalize_loudness(wav)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
