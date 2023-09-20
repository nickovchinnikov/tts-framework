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

