import unittest

import numpy as np
import torch

from training.preprocess.compute_yin import (
    compute_yin,
    cumulativeMeanNormalizedDifferenceFunction,
    differenceFunction,
    getPitch,
    norm_interp_f0,
)


# Create a class to test the ComputePitch class
class TestComputeYin(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_difference_function(self):
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

    def test_get_pitch(self):
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

    def test_cumulative_mean_normalized_difference_function(self):
        # Test the function with a simple example
        df = np.array([1, 2, 3, 4, 5])
        N = len(df)

        # Test the output values
        expected_output = np.array([1.0, 1.0, 1.2, 1.333333, 1.428571])
        actual_output = cumulativeMeanNormalizedDifferenceFunction(df, N)
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)

    def test_compute_yin(self):
        # Test the function with a simple example
        sig = torch.from_numpy(np.sin(2 * np.pi * 440 * np.arange(44100) / 44100))
        sr = 44100

        actual_output = compute_yin(sig, sr)

        # Check the result
        expected_output = np.load("mocks/test_compute_yin.npy")

        np.testing.assert_allclose(actual_output, expected_output)

    def test_norm_interp_f0(self):
        # Test the norm_interp_f0 function with a simple example
        f0 = np.array([0, 100, 0, 200, 0])
        actual_output = norm_interp_f0(f0)
        expected_output = (
            np.array([100, 100, 150, 200, 200]),
            np.array([True, False, True, False, True]),
        )
        np.testing.assert_allclose(actual_output[0], expected_output[0])
        # Test the norm_interp_f0 function with a zero-dimensional array
        np.testing.assert_array_equal(actual_output[1], expected_output[1])
