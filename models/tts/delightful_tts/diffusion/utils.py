from typing import Optional

import torch
from torch import Tensor


# TODO: check this code for integration
def sequence_mask(length: Tensor, max_length: Optional[int] = None) -> Tensor:
    if max_length is None:
        max_length = int(length.max().item())
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)
