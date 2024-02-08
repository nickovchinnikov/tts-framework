import unittest

import numpy as np
import torch

from models.helpers.tools import initialize_embeddings


class TestInitializeEmbeddings(unittest.TestCase):
    def test_initialize_embeddings(self):
        # Test with correct input shape
        shape = (5, 10)
        result = initialize_embeddings(shape)
        # Assert output is torch.Tensor
        self.assertIsInstance(result, torch.Tensor)
        # Assert output shape
        self.assertEqual(result.shape, shape)
        # Assert type of elements
        self.assertEqual(result.dtype, torch.float32)

        # Assert standard deviation is close to expected (within some tolerance)
        expected_stddev = np.sqrt(2 / shape[1])
        tolerance = 0.1
        self.assertLessEqual(abs(result.std().item() - expected_stddev), tolerance)

        # Test with incorrect number of dimensions in shape
        incorrect_shape = (5, 10, 15)
        with self.assertRaises(AssertionError) as context:
            initialize_embeddings(incorrect_shape)
        self.assertEqual(
            str(context.exception), "Can only initialize 2-D embedding matrices ...",
        )

        # Test with zero dimensions in shape
        zero_dim_shape = ()
        with self.assertRaises(AssertionError) as context:
            initialize_embeddings(zero_dim_shape)
        self.assertEqual(
            str(context.exception), "Can only initialize 2-D embedding matrices ...",
        )


# Run tests
if __name__ == "__main__":
    unittest.main()
