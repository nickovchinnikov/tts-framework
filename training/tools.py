from typing import List, Union

import torch
from torch import Tensor, nn


def pad_1D(inputs: List[Tensor], pad_value: float = 0.0) -> Tensor:
    r"""Pad a list of 1D tensor list to the same length.

    Args:
        inputs (List[torch.Tensor]): List of 1D numpy arrays to pad.
        pad_value (float): Value to use for padding. Default is 0.0.

    Returns:
        torch.Tensor: Padded 2D numpy array of shape (len(inputs), max_len), where max_len is the length of the longest input array.
    """
    max_len = max(x.size(0) for x in inputs)
    padded_inputs = [nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_value) for x in inputs]
    return torch.stack(padded_inputs)


def pad_2D(
    inputs: List[Tensor], maxlen: Union[int, None] = None, pad_value: float = 0.0,
) -> Tensor:
    r"""Pad a list of 2D tensor arrays to the same length.

    Args:
        inputs (List[torch.Tensor]): List of 2D numpy arrays to pad.
        maxlen (Union[int, None]): Maximum length to pad the arrays to. If None, pad to the length of the longest array. Default is None.
        pad_value (float): Value to use for padding. Default is 0.0.

    Returns:
        torch.Tensor: Padded 3D numpy array of shape (len(inputs), max_len, input_dim), where max_len is the maximum length of the input arrays, and input_dim is the dimension of the input arrays.
    """
    max_len = max(x.size(1) for x in inputs) if maxlen is None else maxlen

    padded_inputs = [nn.functional.pad(x, (0, max_len - x.size(1), 0, 0), value=pad_value) for x in inputs]
    return torch.stack(padded_inputs)


def pad_3D(inputs: Union[Tensor, List[Tensor]], B: int, T: int, L: int) -> Tensor:
    r"""Pad a 3D torch tensor to a specified shape.

    Args:
        inputs (torch.Tensor): 3D numpy array to pad.
        B (int): Batch size to pad the array to.
        T (int): Time steps to pad the array to.
        L (int): Length to pad the array to.

    Returns:
        torch.Tensor: Padded 3D numpy array of shape (B, T, L), where B is the batch size, T is the time steps, and L is the length.
    """
    if isinstance(inputs, list):
        inputs_padded = torch.zeros(B, T, L, dtype=inputs[0].dtype)
        for i, input_ in enumerate(inputs):
            inputs_padded[i, :input_.size(0), :input_.size(1)] = input_

    elif isinstance(inputs, torch.Tensor):
        inputs_padded = torch.zeros(B, T, L, dtype=inputs.dtype)
        inputs_padded[:inputs.size(0), :inputs.size(1), :inputs.size(2)] = inputs

    return inputs_padded
