from typing import Any, Callable, Dict, Iterable, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from model.config import (
    AcousticModelConfigType,
    AcousticTrainingConfig,
)


def get_lr_lambda(
    model_config: AcousticModelConfigType,
    train_config: AcousticTrainingConfig,
    current_step: int = 0,
) -> Tuple[float, Callable[[int], float]]:
    r"""Returns the custom lambda function for the learning rate schedule.

    Returns
        function: The custom lambda function for the learning rate schedule.
    """
    init_lr = model_config.encoder.n_hidden ** -0.5

    def lr_lambda(step: int = current_step) -> float:
        r"""Computes the learning rate scale factor.

        Args:
            step (int): The current training step.

        Returns:
            float: The learning rate scale factor.
        """
        step = 1 if step == 0 else step

        warmup = train_config.optimizer_config.warm_up_step
        anneal_steps = train_config.optimizer_config.anneal_steps
        anneal_rate = train_config.optimizer_config.anneal_rate

        lr_scale = min(
            step ** -0.5,
            step * warmup ** -1.5,
        )

        for s in anneal_steps:
            if step > s:
                lr_scale *= anneal_rate

        return init_lr * lr_scale

    return init_lr, lr_lambda

class ScheduledOptimPretraining(Optimizer):
    r"""A custom optimizer that uses `AdamW` for optimization and an `LambdaLR` for learning rate scheduling."""

    def __init__(
        self,
        train_config: AcousticTrainingConfig,
        model_config: AcousticModelConfigType,
        parameters: Iterable,
        defaults: Dict[str, Any] = {},
        step: int = 0,
    ):
        r"""Initializes the ScheduledOptimPretraining optimizer.

        Args:
            train_config (AcousticPretrainingConfig): The training configuration.
            model_config (AcousticModelConfigType): The model configuration.
            parameters (Iterable): The model parameters to optimize.
            defaults (Dict[str, Any]): Default optimization options. Defaults to an empty dictionary.
            step (int): The current training step. Defaults to None.
        """
        super().__init__(params=parameters, defaults=defaults)

        init_lr, lr_lambda = get_lr_lambda(
            model_config=model_config,
            train_config=train_config,
            current_step=step,
        )

        self._optimizer = torch.optim.Adam(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
            lr=init_lr,
        )

        self._scheduler = LambdaLR(self._optimizer, lr_lambda)

    def step(self, closure):
        r"""Performs a single optimization step."""
        self._optimizer.step(closure)
        self._scheduler.step()

    def zero_grad(self) -> None:
        r"""Zeroes the gradients of the optimizer."""
        self._optimizer.zero_grad()

    def get_lr(self) -> float:
        r"""Returns the current learning rate."""
        return self._optimizer.param_groups[0]["lr"]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the optimizer state dictionary.

        Args:
            state_dict (Dict[str, Any]): The optimizer state dictionary.
        """
        self._optimizer.load_state_dict(state_dict)
