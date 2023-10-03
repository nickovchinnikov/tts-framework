from typing import List, Union

import numpy as np


def pad_1D(inputs: List[np.ndarray], pad_value: float = 0.0) -> np.ndarray:
    r"""
    Pad a list of 1D numpy arrays to the same length.

    Args:
        inputs (List[np.ndarray]): List of 1D numpy arrays to pad.
        pad_value (float): Value to use for padding. Default is 0.0.

    Returns:
        np.ndarray: Padded 2D numpy array of shape (len(inputs), max_len), where max_len is the length of the longest input array.
    """

    def pad_data(x, length):
        r"""
        Pad a 1D numpy array with zeros to a specified length.

        Args:
            x (np.ndarray): 1D numpy array to pad.
            length (int): Length to pad the array to.

        Returns:
            np.ndarray: Padded 1D numpy array of shape (length,).
        """
        x_padded = np.pad(
            x, (0, length - x.shape[0]), mode="constant", constant_values=pad_value
        )
        return x_padded

    max_len = max(len(x) for x in inputs)
    padded = np.stack([pad_data(x, max_len) for x in inputs])

    return padded


def pad_2D(
    inputs: List[np.ndarray], maxlen: Union[int, None] = None, pad_value: float = 0.0
) -> np.ndarray:
    r"""
    Pad a list of 2D numpy arrays to the same length.

    Args:
        inputs (List[np.ndarray]): List of 2D numpy arrays to pad.
        maxlen (Union[int, None]): Maximum length to pad the arrays to. If None, pad to the length of the longest array. Default is None.
        pad_value (float): Value to use for padding. Default is 0.0.

    Returns:
        np.ndarray: Padded 3D numpy array of shape (len(inputs), max_len, input_dim), where max_len is the maximum length of the input arrays, and input_dim is the dimension of the input arrays.
    """

    def pad(x, max_len):
        r"""
        Pad a 2D numpy array with zeros to a specified length.

        Args:
            x (np.ndarray): 2D numpy array to pad.
            max_len (int): Maximum length to pad the array to.

        Returns:
            np.ndarray: Padded 2D numpy array of shape (x.shape[0], max_len), where x.shape[0] is the number of rows in the input array.
        """
        if np.shape(x)[1] > max_len:
            raise ValueError("not max_len")
        padding = np.ones((x.shape[0], max_len - np.shape(x)[1])) * pad_value
        x = np.concatenate((x, padding), 1)
        return x

    if maxlen:
        output = np.stack([pad(x, maxlen) for x in inputs])
    else:
        max_len = max(np.shape(x)[1] for x in inputs)
        output = np.stack([pad(x, max_len) for x in inputs])
    return output


def pad_3D(inputs: List[np.ndarray], B: int, T: int, L: int) -> np.ndarray:
    r"""
    Pad a 3D numpy array to a specified shape.

    Args:
        inputs (np.ndarray): 3D numpy array to pad.
        B (int): Batch size to pad the array to.
        T (int): Time steps to pad the array to.
        L (int): Length to pad the array to.

    Returns:
        np.ndarray: Padded 3D numpy array of shape (B, T, L), where B is the batch size, T is the time steps, and L is the length.
    """
    inputs_padded = np.zeros((B, T, L), dtype=np.float32)
    for i, input_ in enumerate(inputs):
        inputs_padded[i, : np.shape(input_)[0], : np.shape(input_)[1]] = input_
    return inputs_padded
