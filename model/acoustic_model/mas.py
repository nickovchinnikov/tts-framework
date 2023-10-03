import numpy as np
from numba import jit, njit, prange

# TODO:
# Check this: https://github.com/archinetai/aligner-pytorch
# and this: https://github.com/resemble-ai/monotonic_align/tree/master


# TODO: Don't see any performance improvement with numba
@njit(fastmath=True)
def mas_width1(attn_map: np.ndarray) -> np.ndarray:
    r"""
    Applies a Monotonic Alignments Shrink (MAS) operation with a hard-coded width of 1 to an attention map.
    Mas with hardcoded width=1
    Essentially, it produces optimal alignments based on previous attention distribution.

    Args:
        attn_map (np.ndarray): The original attention map, a 2D numpy array where rows correspond to mel bins and columns to text bins.

    Returns:
        opt (np.ndarray): Returns the optimal attention map after applying the MAS operation.
    """
    # assumes mel x text
    # Create a placeholder for the output
    opt = np.zeros_like(attn_map)

    # Convert the attention map to log scale for stability
    attn_map = np.log(attn_map)

    # Initialize the first row of attention map appropriately
    attn_map[0, 1:] = -np.inf

    # Initialize log_p with the first row of attention map
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]

    # Placeholder to remember the previous indices for backtracking later
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)

    # Compute the log probabilities based on previous attention distribution
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            # Compare with left (j-1) pixel and update if the left pixel has larger log probability
            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log

            # Store the position of maximum cumulative log probability
            prev_ind[i, j] = prev_j

    # Backtrack to retrieve the path of attention with maximum cumulative log probability
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]

    # Mark the first position of the optimal path
    opt[0, curr_text_idx] = 1
    return opt


@njit(parallel=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    r"""
    Applies Monotonic Alignments Shrink (MAS) operation in parallel to the batches of an attention map.
    It uses the `mas_width1` function internally to perform MAS operation.

    Args:
        b_attn_map (np.ndarray): The batched attention map; a 3D array where the first dimension is the batch size, second dimension corresponds to source length, and third dimension corresponds to target length.
        in_lens (np.ndarray): Lengths of sequences in the input batch.
        out_lens (np.ndarray): Lengths of sequences in the output batch.
        width (int, optional): The width for the MAS operation. Defaults to 1.

    Raises:
        AssertionError: If width is not equal to 1. This function currently supports only width of 1.

    Returns:
        np.ndarray: The batched attention map after applying the MAS operation. It has the same dimensions as `b_attn_map`.
    """
    # Assert that the width is 1. This function currently supports only width of 1
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    # Loop over each attention map in the batch in parallel
    for b in prange(b_attn_map.shape[0]):
        # Apply Monotonic Alignments Shrink operation to the b-th attention map in the batch
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])

        # Update the b-th attention map in the output with the result of MAS operation
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out

    # Return the batched attention map after applying the MAS operation
    return attn_out
