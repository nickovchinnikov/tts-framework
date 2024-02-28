import torch
from torch import Tensor


def sample_wise_min_max(x: Tensor) -> Tensor:
    r"""Applies sample-wise min-max normalization to a tensor.

    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, num_samples, num_features).

    Returns:
        torch.Tensor: Normalized tensor of the same shape as the input tensor.
    """
    # Compute the maximum and minimum values of each sample in the batch
    maximum = torch.amax(x, dim=(1, 2), keepdim=True)
    minimum = torch.amin(x, dim=(1, 2), keepdim=True)

    # Apply sample-wise min-max normalization to the input tensor
    return (x - minimum) / (maximum - minimum)
