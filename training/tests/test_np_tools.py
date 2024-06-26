import unittest

import numpy as np

from training.np_tools import pad_1D, pad_2D, pad_3D


class TestPad(unittest.TestCase):
    def test_pad_1D(self):
        # Test case 1: Pad a list of 1D numpy arrays with different lengths
        inputs = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])]
        expected_output = np.array([[1, 2, 3], [4, 5, 0], [6, 0, 0]])
        self.assertTrue(np.array_equal(pad_1D(inputs), expected_output))

        # Test case 2: Pad a list of 1D numpy arrays with the same length
        inputs = [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])]
        expected_output = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.assertTrue(np.array_equal(pad_1D(inputs), expected_output))

        # Test case 3: Pad a list of 1D numpy arrays with a non-zero pad value
        inputs = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8, 9])]
        expected_output = np.array([[1, 2, 0, 0], [3, 4, 5, 0], [6, 7, 8, 9]])
        self.assertTrue(np.array_equal(pad_1D(inputs, pad_value=0.0), expected_output))

        # Test case 4: Pad a list of 1D numpy arrays with a non-zero pad value
        inputs = [np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8, 9])]
        expected_output = np.array([[1, 2, 1, 1], [3, 4, 5, 1], [6, 7, 8, 9]])
        self.assertTrue(np.array_equal(pad_1D(inputs, pad_value=1.0), expected_output))

        # Test case 5: Pad a list of 1D numpy arrays with a single non-empty array
        inputs = [np.array([1, 2, 3])]
        expected_output = np.array([[1, 2, 3]])
        self.assertTrue(np.array_equal(pad_1D(inputs), expected_output))

    def test_pad_2D(self):
        # Test case 1: Pad a list of 2D numpy arrays with different shapes
        inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]])]
        expected_output = np.array([[[1, 2, 0], [3, 4, 0]], [[5, 6, 7], [8, 9, 10]]])
        self.assertTrue(np.array_equal(pad_2D(inputs), expected_output))

        # Test case 2: Pad a list of 2D numpy arrays with the same shape
        inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])]
        expected_output = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        self.assertTrue(np.array_equal(pad_2D(inputs), expected_output))

        # Test case 3: Pad a list of 2D numpy arrays with a non-zero pad value
        inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]])]
        expected_output = np.array([[[1, 2, 1], [3, 4, 1]], [[5, 6, 7], [8, 9, 10]]])
        self.assertTrue(np.array_equal(pad_2D(inputs, pad_value=1.0), expected_output))

        # Test case 4: Pad a list of 2D numpy arrays with a maximum length
        inputs = [np.array([[1, 2], [3, 4]]), np.array([[5, 6, 7], [8, 9, 10]])]
        expected_output = np.array([[[1, 2, 0], [3, 4, 0]], [[5, 6, 7], [8, 9, 10]]])
        self.assertTrue(np.array_equal(pad_2D(inputs, maxlen=3), expected_output))

        # Test case 5: Pad a list of 2D numpy arrays with a single non-empty array
        inputs = [np.array([[1, 2], [3, 4]])]
        expected_output = np.array([[[1, 2], [3, 4]]])
        self.assertTrue(np.array_equal(pad_2D(inputs), expected_output))

    def test_pad_3D(self):
        # Test case 1: Pad a 3D numpy array with different dimensions
        inputs = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        expected_output = np.array(
            [
                [[1, 2, 0], [3, 4, 0], [0, 0, 0]],
                [[5, 6, 0], [7, 8, 0], [0, 0, 0]],
                [[9, 10, 0], [11, 12, 0], [0, 0, 0]],
            ],
        )
        self.assertTrue(np.array_equal(pad_3D(inputs, B=3, T=3, L=3), expected_output))

        # Test case 2: Pad a 3D numpy array with the same dimensions
        inputs = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]])
        expected_output = np.array(
            [
                [[1, 2], [3, 4]],
                [[5, 6], [7, 8]],
                [[9, 10], [11, 12]],
            ],
        )
        self.assertTrue(np.array_equal(pad_3D(inputs, B=3, T=2, L=2), expected_output))

        # Test case 3: Pad a 3D numpy array with a single element
        inputs = np.array([[[1, 2], [3, 4]]])
        expected_output = np.array([[[1, 2, 0], [3, 4, 0]]])
        self.assertTrue(np.array_equal(pad_3D(inputs, B=1, T=2, L=3), expected_output))


if __name__ == "__main__":
    unittest.main()
