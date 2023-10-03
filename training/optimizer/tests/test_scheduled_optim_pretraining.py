import unittest

import numpy as np
import torch
from torch import nn

from model.config import AcousticENModelConfig, AcousticPretrainingConfig
from training.optimizer import ScheduledOptimPretraining


class TestScheduledOptimPretraining(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")

        self.parameters = nn.ParameterList(
            [nn.Parameter(torch.randn((10, 10), device=self.device))]
        )

        self.train_config = AcousticPretrainingConfig()

        self.model_config = AcousticENModelConfig()

        self.current_step = 10
        self.optimizer = ScheduledOptimPretraining(
            parameters=self.parameters,
            train_config=self.train_config,
            model_config=self.model_config,
            current_step=self.current_step,
        )

    def test_get_lr_scale(self):
        # Call the _get_lr_scale method
        lr_scale = self.optimizer._get_lr_scale()

        self.assertAlmostEqual(lr_scale, 3.952847075210474e-05, places=10)

    def test_update_learning_rate(self):
        # Call the _update_learning_rate method
        self.optimizer._update_learning_rate(step=1000)

        self.assertAlmostEqual(
            self.optimizer._optimizer.param_groups[0]["lr"],
            0.00020171788261496966,
            places=10,
        )

    def test_step_and_update_lr(self):
        # Call the step_and_update_lr method
        self.optimizer.step_and_update_lr(step=1000)

        # Check that the learning rate was updated correctly
        self.assertAlmostEqual(
            self.optimizer._optimizer.param_groups[0]["lr"],
            0.00020171788261496966,
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
                }
            ],
        }

        # Call the load_state_dict method
        self.optimizer._optimizer.load_state_dict(state_dict)

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
