import torch
import torch.nn.functional as F

from typing import List


def pad(input_ele: List[torch.Tensor], max_len: int) -> torch.Tensor:
    r"""
    Takes a list of 1D or 2D tensors and pads them to match the maximum length.

    Args:
        input_ele (List[torch.Tensor]): The list of tensors to be padded.
        max_len (int): The length to which the tensors should be padded.

    Returns:
        torch.Tensor: A tensor containing all the padded input tensors.
    """

    # Create an empty list to store the padded tensors
    out_list = torch.jit.annotate(List[torch.Tensor], [])
    for batch in input_ele:
        if len(batch.shape) == 1:
            # Perform padding for 1D tensor
            one_batch_padded = F.pad(
                batch, (0, max_len - batch.size(0)), "constant", 0.0
            )
        else:
            # Perform padding for 2D tensor
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0
            )
        # Append the padded tensor to the list
        out_list.append(one_batch_padded)

    # Stack all the tensors in the list into a single tensor
    out_padded = torch.stack(out_list)
    return out_padded


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    r"""
    Generate a mask tensor from a tensor of sequence lengths.

    Args:
        lengths (torch.Tensor): A tensor of sequence lengths of shape: (batch_size, )

    Returns:
        torch.Tensor: A mask tensor of shape: (batch_size, max_len) where max_len is the
            maximum sequence length in the provided tensor. The mask tensor has a value of
            True at each position that is more than the length of the sequence (padding positions).

    Example:
      lengths: `torch.tensor([2, 3, 1, 4])`
      Mask tensor will be: `torch.tensor([
            [False, False, True, True],
            [False, False, False, True],
            [False, True, True, True],
            [False, False, False, False]
        ])`
    """
    # Get batch size
    batch_size = lengths.shape[0]

    # Get maximum sequence length in the batch
    max_len = int(torch.max(lengths).item())

    # Generate a tensor of shape (batch_size, max_len)
    # where each row contains values from 0 to max_len
    ids = (
        torch.arange(0, max_len, device=lengths.device)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    # Compare each value in the ids tensor with
    # corresponding sequence length to generate a mask.
    # The mask will have True at positions where id >= sequence length,
    # indicating padding positions in the original sequences
    mask = ids >= lengths.unsqueeze(1).type(torch.int64).expand(-1, max_len)
    return mask


def stride_lens_downsampling(lens: torch.Tensor, stride: int = 2) -> torch.Tensor:
    r"""
    This function computes the lengths of 1D tensor when applying a stride for downsampling.

    Args:
        lens (torch.Tensor): Tensor containing the lengths to be downsampled.
        stride (int, optional): The stride to be used for downsampling. Defaults to 2.

    Returns:
        torch.Tensor: A tensor of the same shape as the input containing the downsampled lengths.
    """
    # The torch.ceil function is used to handle cases where the length is not evenly divisible
    # by the stride. The torch.ceil function rounds up to the nearest integer, ensuring that
    # each item is present at least once in the downsampled lengths.
    # Finally, the .int() is used to convert the resulting float32 tensor to an integer tensor.
    return torch.ceil(lens / stride).int()
