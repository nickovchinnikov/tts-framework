from typing import Any, Dict, Iterable

import numpy as np
import torch

from model.config import (
    AcousticModelConfigType,
    AcousticTrainingConfig,
)


class ScheduledOptimPretraining:
    def __init__(
        self,
        parameters: Iterable,
        train_config: AcousticTrainingConfig,
        model_config: AcousticModelConfigType,
        current_step: int,
    ):
        r"""Initializes the ScheduledOptimPretraining optimizer.

        Args:
            parameters (Iterable): The model parameters to optimize.
            train_config (AcousticPretrainingConfig): The training configuration.
            model_config (AcousticModelConfigType): The model configuration.
            current_step (int): The current training step.
        """
        self._optimizer = torch.optim.Adam(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
        )
        self.n_warmup_steps = train_config.optimizer_config.warm_up_step
        self.anneal_steps = train_config.optimizer_config.anneal_steps
        self.anneal_rate = train_config.optimizer_config.anneal_rate
        self.current_step = current_step
        self.init_lr = model_config.encoder.n_hidden**-0.5

    def step_and_update_lr(self, step: int) -> None:
        r"""Updates the learning rate and takes a step of the optimizer.

        Args:
            step (int): The current training step.
        """
        self._update_learning_rate(step)
        self._optimizer.step()

    def zero_grad(self) -> None:
        r"""Zeroes the gradients of the optimizer."""
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the optimizer state dictionary.

        Args:
            state_dict (Dict[str, Any]): The optimizer state dictionary.
        """
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self) -> float:
        r"""Computes the learning rate scale factor.

        Returns
            float: The learning rate scale factor.
        """
        lr_scale = np.min(
            [
                np.power(1 if self.current_step == 0 else self.current_step, -0.5),
                np.power(self.n_warmup_steps, -1.5) * self.current_step,
            ],
        )
        for s in self.anneal_steps:
            if self.current_step > s:
                lr_scale = lr_scale * self.anneal_rate
        return lr_scale

    def _update_learning_rate(self, step: int) -> None:
        r"""Updates the learning rate based on the current step.

        Args:
            step (int): The current training step.
        """
        self.current_step = step
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
