import math
import unittest

import numpy as np
import torch

from training.preprocess.tools import (
    byte_encode,
    compute_yin,
    cumulativeMeanNormalizedDifferenceFunction,
    differenceFunction,
    getPitch,
    preprocess_audio,
    resample,
    stereo_to_mono,
)


# Create a class to test the ComputePitch class
class TestTools(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_differenceFunction(self):
        # Test the differenceFunction function with a simple example
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        N = len(x)
        tau_max = 3

        # Test the output shape
        expected_shape = (tau_max,)
        actual_shape = differenceFunction(x, N, tau_max).shape

        self.assertTrue(actual_shape == expected_shape)

        # Test the output values
        expected_output = np.array([0.0, 4.0, 12.0])
        actual_output = differenceFunction(x, N, tau_max)

        np.testing.assert_equal(actual_output, expected_output)

        # Test the function with a larger example
        x = np.random.randn(1000)
        N = len(x)
        tau_max = 100

        # Test the output shape
        expected_shape = (tau_max,)
        actual_shape = differenceFunction(x, N, tau_max).shape

        self.assertTrue(actual_shape == expected_shape)

        # Test the output values
        expected_output = np.zeros(tau_max)
        for tau in range(1, tau_max):
            expected_output[tau] = np.sum((x[: N - tau] - x[tau:N]) ** 2)

        actual_output = differenceFunction(x, N, tau_max)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

    def test_getPitch(self):
        # Test the getPitch function with a simple example
        cmdf = np.array([1.0, 0.5, 0.2, 0.1, 0.05])
        tau_min = 1
        tau_max = 5
        harmo_th = 0.3

        # Test the output value when there is a value below the threshold
        expected_output = 4
        actual_output = getPitch(cmdf, tau_min, tau_max, harmo_th)
        self.assertEqual(actual_output, expected_output)

        # Test the output value when there are no values below the threshold
        cmdf = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
        expected_output = 0
        actual_output = getPitch(cmdf, tau_min, tau_max, harmo_th)
        self.assertEqual(actual_output, expected_output)

        # Test the output value when there are no values below the threshold
        cmdf = np.random.rand(100)
        cmdf[tau_min:tau_max] = 0.5
        expected_output = 0
        actual_output = getPitch(cmdf, tau_min, tau_max, harmo_th)
        self.assertEqual(actual_output, expected_output)

    def test_cumulativeMeanNormalizedDifferenceFunction(self):
        # Test the function with a simple example
        df = np.array([1, 2, 3, 4, 5])
        N = len(df)

        # Test the output values
        expected_output = np.array([1.0, 1.0, 1.2, 1.333333, 1.428571])
        actual_output = cumulativeMeanNormalizedDifferenceFunction(df, N)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

    def test_compute_yin(self):
        # Test the function with a simple example
        sig = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
        sr = 44100

        actual_output = compute_yin(sig, sr)

        # Check the result
        expected_output = np.load("mocks/test_compute_yin.npy")

        np.testing.assert_allclose(actual_output, expected_output)

    def test_byte_encode(self):
        # Test with a simple word
        word = "hello"
        expected_output = [104, 101, 108, 108, 111]
        self.assertTrue(byte_encode(word) == expected_output)

        # Test with a word containing non-ASCII characters
        word = "héllo"
        expected_output = [104, 195, 169, 108, 108, 111]
        self.assertTrue(byte_encode(word) == expected_output)

        # Test with an empty string
        word = ""
        expected_output = []
        self.assertTrue(byte_encode(word) == expected_output)

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

        # Test the output values
        expected_output = np.array([0.62589299, 3.51319581, 0.0])
        actual_output = resample(wav, orig_sr, target_sr)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

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
        expected_output = torch.tensor([2.434131, 0.000000])
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)
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


if __name__ == "__main__":
    unittest.main()
