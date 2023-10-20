import unittest

import torch
from torch import nn

from model.config import AcousticENModelConfig, AcousticPretrainingConfig
from training.optimizer.scheduled_optim_pretraining import (
    ScheduledOptimPretraining,
    get_lr_lambda,
)


class TestScheduledOptimPretraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        self.parameters = nn.ParameterList(
            [nn.Parameter(torch.randn((10, 10), device=self.device))],
        )

        self.train_config = AcousticPretrainingConfig()

        self.model_config = AcousticENModelConfig()

        self.init_lr = self.model_config.encoder.n_hidden ** -0.5

        self.current_step = 10
        self.optimizer = ScheduledOptimPretraining(
            parameters=self.parameters,
            train_config=self.train_config,
            model_config=self.model_config,
            step=self.current_step,
        )

    def test_get_lr_lambda(self):
        current_step = 5000

        init_lr, lr_lambda = get_lr_lambda(
            self.model_config,
            self.train_config,
            current_step,
        )

        # Test the returned function
        self.assertEqual(init_lr, self.init_lr)
        self.assertAlmostEqual(lr_lambda(0), 2.0171788261496964e-07, places=10)
        self.assertAlmostEqual(lr_lambda(10), 2.0171788261496965e-06, places=10)
        self.assertAlmostEqual(lr_lambda(100), 2.0171788261496963e-05, places=10)
        self.assertAlmostEqual(lr_lambda(1000), 0.00020171788261496966, places=10)
        self.assertAlmostEqual(lr_lambda(current_step), 0.0007216878364870322, places=10)


    def test_initial_lr(self):
        lr = self.optimizer.get_lr()

        self.assertAlmostEqual(lr, 1.0293872591693944e-08, places=10)

    def test_step_and_update_lr(self):
        lr = self.optimizer.get_lr()

        self.assertAlmostEqual(lr, 1.0293872591693944e-08, places=10)

        for _ in range(100):
            self.optimizer.step()

        lr = self.optimizer.get_lr()
        expected_lr = 1.0293872591693944e-06

        self.assertAlmostEqual(
            lr,
            expected_lr,
            places=10,
        )

        for _ in range(1000):
            self.optimizer.step()

        lr = self.optimizer.get_lr()
        expected_lr = 1.1323259850863337e-05

        self.assertAlmostEqual(
            lr,
            expected_lr,
            places=10,
        )

    def test_zero_grad(self):
        # Call the zero_grad method
        self.optimizer.zero_grad()

        # Check that the gradients were zeroed
        for param in self.parameters:
            self.assertTrue(param.grad is None)

    def test_load_state_dict(self):
        # Define a mock state dictionary
        state_dict = {
            "state": {},
            "param_groups": [
                {
                    "lr": 0.05,
                    "betas": (0.1, 0.8),
                    "eps": 1e-07,
                    "weight_decay": 1,
                    "amsgrad": True,
                    "maximize": True,
                    "foreach": None,
                    "capturable": True,
                    "differentiable": True,
                    "fused": None,
                    "params": [0],
                },
            ],
        }

        # Call the load_state_dict method
        self.optimizer.load_state_dict(state_dict)

        # Get the actual state dictionary
        actual_state_dict = self.optimizer._optimizer.state_dict()

        keys_to_compare = state_dict["param_groups"][0].keys()

        # Create new dictionaries that contain only the param_groups keys and values
        expected_param_groups = {
            k: v
            for k, v in state_dict["param_groups"][0].items()
            if k in keys_to_compare
        }
        actual_param_groups = {
            k: v
            for k, v in actual_state_dict["param_groups"][0].items()
            if k in keys_to_compare
        }

        # Compare the param_groups dictionaries
        self.assertDictEqual(expected_param_groups, actual_param_groups)


if __name__ == "__main__":
    unittest.main()
