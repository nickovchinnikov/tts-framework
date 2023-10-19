from typing import Any, Dict, Iterable

import torch

from model.config import AcousticTrainingConfig


class ScheduledOptimFinetuning:
    def __init__(
        self,
        parameters: Iterable,
        train_config: AcousticTrainingConfig,
        current_step: int,
    ):
        r"""Initializes a ScheduledOptimFinetuning optimizer.

        Args:
        ----
            parameters (Iterable): Iterable of model parameters to optimize.
            train_config (AcousticTrainingConfig): Configuration object for the optimizer.
            current_step (int): Current training step.
        """
        self._optimizer = torch.optim.AdamW(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
        )
        self.current_step = current_step
        self.init_lr = train_config.optimizer_config.learning_rate
        self.lr_decay = train_config.optimizer_config.lr_decay

    def step_and_update_lr(self, step: int) -> None:
        r"""Updates the learning rate and takes a step of the optimizer.

        Args:
        ----
            step (int): Current training step.
        """
        self._update_learning_rate(step)
        self._optimizer.step()

    def zero_grad(self) -> None:
        r"""Zeroes the gradients of all optimized parameters."""
        self._optimizer.zero_grad()

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the optimizer state dictionary.

        Args:
        ----
            state_dict (Dict[str, Any]): State dictionary to load.
        """
        self._optimizer.load_state_dict(state_dict)

    def _get_lr_scale(self) -> float:
        r"""Computes the learning rate scale factor.

        Returns
        -------
            float: Learning rate scale factor.
        """
        return self.lr_decay**self.current_step

    def _update_learning_rate(self, step: int) -> None:
        r"""Updates the learning rate based on the current step.

        Args:
        ----
            step (int): Current training step.
        """
        self.current_step = step
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
