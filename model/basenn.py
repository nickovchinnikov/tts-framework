from typing import Any

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule


class BaseNNModule(LightningModule):
    r"""
    This is the base class for all neural networks in this module. It provides basic support for device
    placement of the model.

    In PyTorch, a significant feature is the ability to move your entire model to a GPU for better
    performance. All subclasses should ensure to forward the device argument to this class's constructor
    to ensure that all of its parameters and buffers are moved.

    Examples:
    ```python
    model = BaseNN()
    ```
    """

    def __init__(self):
        super().__init__()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)
