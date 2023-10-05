from typing import Any

import torch
import torch.nn as nn

from model.helpers.tools import get_device


class BaseNNModule(nn.Module):
    r"""
    This is the base class for all neural networks in this module. It provides basic support for device
    placement of the model.

    In PyTorch, a significant feature is the ability to move your entire model to a GPU for better
    performance. All subclasses should ensure to forward the device argument to this class's constructor
    to ensure that all of its parameters and buffers are moved.

    Args:
        device (torch.device): The device to which the model should be moved. Defaults `get_device()`

    Attributes:
        device (torch.device): The device to which the model is currently allocated.

    Examples:
    ```python
    model = BaseNN(device=torch.device('cpu'))
    ```
    """

    def __init__(
        self,
        device: torch.device = get_device(),
    ):
        super().__init__()

        # Store the device with the model for later reference
        self.device = device

    # Removed __call__ method as pl.LightningModule handles device placement
