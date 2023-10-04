import os
import unittest

import numpy as np

from training.preprocess.tools import (
    compute_yin,
    cumulativeMeanNormalizedDifferenceFunction,
    differenceFunction,
    getPitch,
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

        # Test the function with a larger example
        cmdf = np.random.rand(100)
        tau_min = 10
        tau_max = 50
        harmo_th = 0.1

        # Test the output value when there is a value below the threshold
        expected_output = np.argmin(cmdf[tau_min:tau_max]) + tau_min
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

        # Test the output values
        expected_output = (
            np.array([440.0] * 171),
            [0.0] * 171,
            [0.0] * 171,
            [t / float(sr) for t in range(0, len(sig) - 512, 256)],
        )
        actual_output = compute_yin(sig, sr)

        # Check the result
        expected_output = np.load(
            os.path.abspath("training/preprocess/tests/test_compute_yin.npy")
        )

        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
