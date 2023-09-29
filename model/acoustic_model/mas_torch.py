import torch


def mas_width1(attn_map):
    # assumes mel x text
    # Create a placeholder for the output
    opt = torch.zeros_like(attn_map)

    # Convert the attention map to log scale for stability
    attn_map = torch.log(attn_map)

    # Initialize the first row of attention map appropriately
    attn_map[0, 1:] = -float("inf")

    # Initialize log_p with the first row of attention map
    log_p = torch.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]

    # Placeholder to remember the previous indices for backtracking later
    prev_ind = torch.zeros_like(attn_map, dtype=torch.int64)

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


def mas_width1_(attn_map):
    # assumes mel x text
    # Create a placeholder for the output
    opt = torch.zeros_like(attn_map)

    # Convert the attention map to log scale for stability
    attn_map = torch.log(attn_map)

    # Initialize the first row of attention map appropriately
    attn_map[0, 1:] = -float("inf")

    # Initialize log_p with the first row of attention map
    log_p = torch.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]

    # Compute the log probabilities based on previous attention distribution
    for i in range(1, attn_map.shape[0]):
        # Compute the maximum log probability for each text dimension
        max_log_p = torch.max(log_p[i - 1, : attn_map.shape[1] - 1], log_p[i - 1, 1:])
        max_log_p = torch.cat(
            [max_log_p[0].unsqueeze(0), max_log_p, max_log_p[-1].unsqueeze(0)], dim=0
        )

        # Compute the indices of the maximum log probability for each text dimension
        prev_ind = torch.argmax(max_log_p, dim=0) - 1

        # Compute the log probabilities for the current row
        log_p[i, :] = (
            attn_map[i, :] + max_log_p[prev_ind + 1, torch.arange(attn_map.shape[1])]
        )

        # Store the position of maximum cumulative log probability
        prev_ind = torch.clamp(prev_ind, 0, attn_map.shape[1] - 1)
        opt[i, :] = opt[i - 1, prev_ind]

    # Mark the first position of the optimal path
    opt[0, torch.argmax(attn_map[0, :])] = 1
    return opt


def b_mas(b_attn_map, in_lens, out_lens, width=1):
    # Assert that the width is 1. This function currently supports only width of 1
    assert width == 1
    attn_out = torch.zeros_like(b_attn_map)

    # Loop over each attention map in the batch
    for b in range(b_attn_map.shape[0]):
        # Apply Monotonic Alignments Shrink operation to the b-th attention map in the batch
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])

        # Update the b-th attention map in the output with the result of MAS operation
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out

    # Return the batched attention map after applying the MAS operation
    return attn_out
