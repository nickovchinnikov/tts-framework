import unittest

import numpy as np
import torch

from models.enhancer.gaussian_diffusion.utils import (
    default,
    exists,
    extract,
    get_noise_schedule_list,
    noise_like,
    vpsde_beta_t,
)


class TestNoiseSchedule(unittest.TestCase):
    def test_exists(self):
        # Test when input is not None
        self.assertTrue(exists(5))
        self.assertTrue(exists(0))
        self.assertTrue(exists([]))
        self.assertTrue(exists({}))
        self.assertTrue(exists("Hello"))

        # Test when input is None
        self.assertFalse(exists(None))

    def test_default(self):
        # Test when input value exists
        self.assertEqual(default(5, 10), 5)
        self.assertEqual(default(0, 10), 0)
        self.assertEqual(default([], [1, 2]), [])
        self.assertEqual(default(None, [1, 2]), [1, 2])
        self.assertEqual(default("Hello", "World"), "Hello")

        # Test when input value does not exist
        self.assertEqual(default(None, 10), 10)
        self.assertEqual(default(None, lambda: 10), 10)
        self.assertEqual(default(None, "Default"), "Default")

    def test_extract(self):
        # Test case 1
        a = torch.tensor([[1, 2], [3, 4], [5, 6]])
        t = torch.tensor([[0], [1], [0]])
        x_shape = (3, 2)

        expected_output = torch.tensor([[1], [4], [5]])
        self.assertTrue(torch.allclose(extract(a, t, x_shape), expected_output))

    def test_noise_like(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Test case without repeat
        shape = (3, 2)
        noise = noise_like(shape, device)
        self.assertEqual(noise.shape, shape)

        # Test case with repeat
        shape = (3, 2)
        noise = noise_like(shape, device, repeat=True)
        self.assertEqual(noise.shape, shape)

    def test_vpsde_beta_t(self):
        min_beta = 0.1
        max_beta = 0.5
        T = 10
        t = 5
        beta = vpsde_beta_t(t, T, min_beta, max_beta)
        self.assertIsInstance(beta, float, "Beta coefficient should be a float.")
        self.assertTrue(0 <= beta <= 1, "Beta coefficient should be between 0 and 1.")

    def test_get_noise_schedule_list(self):
        timesteps = 10
        linear_schedule = get_noise_schedule_list("linear", timesteps)
        self.assertIsInstance(linear_schedule, np.ndarray, "Linear schedule should be a numpy array.")
        self.assertEqual(len(linear_schedule), timesteps, "Linear schedule length should match timesteps.")

        cosine_schedule = get_noise_schedule_list("cosine", timesteps)
        self.assertIsInstance(cosine_schedule, np.ndarray, "Cosine schedule should be a numpy array.")
        self.assertEqual(len(cosine_schedule), timesteps, "Cosine schedule length should match timesteps.")

        vpsde_schedule = get_noise_schedule_list("vpsde", timesteps)
        self.assertIsInstance(vpsde_schedule, np.ndarray, "VPSDE schedule should be a numpy array.")
        self.assertEqual(len(vpsde_schedule), timesteps, "VPSDE schedule length should match timesteps.")

        with self.assertRaises(NotImplementedError):
            invalid_schedule = get_noise_schedule_list("invalid_mode", timesteps)

if __name__ == "__main__":
    unittest.main()
