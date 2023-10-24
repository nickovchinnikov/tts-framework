from typing import Any, Dict, Iterable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR

from model.config import AcousticTrainingConfig


class ScheduledOptimFinetuning(Optimizer):
    r"""A custom optimizer that uses `AdamW` for optimization and an `ExponentialLR` for learning rate scheduling.

    Args:
        train_config (AcousticTrainingConfig): Training configuration with optimizer and scheduler parameters.
        parameters (Iterable): Iterable of parameters to optimize.
        defaults (Dict[str, Any]): Default optimization options. Defaults to an empty dictionary.
        step (Optional[int]): The current training step. Defaults to None.
    """

    def __init__(
        self,
        train_config: AcousticTrainingConfig,
        parameters: Iterable,
        defaults: Dict[str, Any] = {},
        step: Optional[int] = None,
    ):
        super().__init__(params=parameters, defaults=defaults)

        # Compute the gamma and initial learning rate based on the current step
        lr_decay = train_config.optimizer_config.lr_decay
        default_lr = train_config.optimizer_config.learning_rate

        init_lr = default_lr if step is None else default_lr * (lr_decay ** step)

        self._optimizer = torch.optim.AdamW(
            parameters,
            betas=train_config.optimizer_config.betas,
            eps=train_config.optimizer_config.eps,
            lr=init_lr,
        )

        self._scheduler = ExponentialLR(self._optimizer, gamma=lr_decay)

    def step(self, closure):
        r"""Performs a single optimization step."""
        self._optimizer.step(closure)
        self._scheduler.step()

    def zero_grad(self) -> None:
        r"""Clears the gradients of all optimized parameters.
        This should be called before the backward pass in PyTorch.
        """
        self._optimizer.zero_grad()

    def get_lr(self) -> float:
        r"""Returns the current learning rate."""
        return self._optimizer.param_groups[0]["lr"]

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Loads the optimizer state.

        Args:
        state_dict (Dict[str, Any]): A dictionary containing a whole state of the optimizer.
        """
        self._optimizer.load_state_dict(state_dict)
