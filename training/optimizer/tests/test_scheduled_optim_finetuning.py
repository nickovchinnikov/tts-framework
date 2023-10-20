import unittest

import torch
from torch import nn

from model.config import AcousticFinetuningConfig
from training.optimizer import ScheduledOptimFinetuning


class TestScheduledOptimFinetuning(unittest.TestCase):
    def setUp(self):
        self.train_config = AcousticFinetuningConfig()
        self.learning_rate = self.train_config.optimizer_config.learning_rate
        self.lr_decay = self.train_config.optimizer_config.lr_decay

        self.parameters = nn.ParameterList(
            [nn.Parameter(torch.randn((10, 10)))],
        )

        self.optimizer = ScheduledOptimFinetuning(
            parameters=self.parameters,
            train_config=self.train_config,
        )

    def test_initial_lr(self):
        lr = self.optimizer.get_lr()

        self.assertEqual(lr, self.learning_rate)

    def test_step(self):
        lr = self.optimizer.get_lr()

        self.assertEqual(lr, self.learning_rate)

        self.optimizer.step()
        lr = self.optimizer.get_lr()
        expected_lr = self.learning_rate * self.lr_decay

        self.assertAlmostEqual(
            lr,
            expected_lr,
            places=10,
        )

    def test_initial_step_prop_and_step_action(self):
        # Create a new optimizer with a step of 1000
        initial_step = 1000
        optimizer = ScheduledOptimFinetuning(
            parameters=self.parameters,
            train_config=self.train_config,
            step=initial_step,
        )

        lr = optimizer.get_lr()
        expected_lr = self.learning_rate * (self.lr_decay ** initial_step)

        # Check that the learning rate was updated correctly
        self.assertAlmostEqual(
            lr,
            expected_lr,
            places=10,
        )

        optimizer.step()
        lr = optimizer.get_lr()

        expected_lr = expected_lr * self.lr_decay

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
