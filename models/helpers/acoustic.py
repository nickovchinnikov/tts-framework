import math

import torch


def positional_encoding(
    d_model: int, length: int,
) -> torch.Tensor:
    r"""Function to calculate positional encoding for transformer model.

    Args:
        d_model (int): Dimension of the model (often corresponds to embedding size).
        length (int): Length of sequences.

    Returns:
        torch.Tensor: Tensor having positional encodings.
    """
    # Initialize placeholder for positional encoding
    pe = torch.zeros(length, d_model)

    # Generate position indices and reshape to have shape (length, 1)
    position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)

    # Calculate term for division
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * -(math.log(10000.0) / d_model),
    )

    # Assign sin of position * div_term to even indices in the encoding matrix
    pe[:, 0::2] = torch.sin(position * div_term)

    # Assign cos of position * div_term to odd indices in the encoding matrix
    pe[:, 1::2] = torch.cos(position * div_term)

    # Add an extra dimension to match expected output shape
    return pe.unsqueeze(0)


def pitch_phoneme_averaging(
        durations: torch.Tensor,
        pitches: torch.Tensor,
        max_phoneme_len: int) -> torch.Tensor:
    r"""Function to compute the average pitch values over the duration of each phoneme.

    Args:
        durations (torch.Tensor): Duration of each phoneme for each sample in a batch.
                                  Shape: (batch_size, n_phones)
        pitches (torch.Tensor): Per-frame pitch values for each sample in a batch.
                                Shape: (batch_size, n_mel_timesteps)
        max_phoneme_len (int): Maximum length of the phoneme sequence in a batch.

    Returns:
        pitches_averaged (torch.Tensor): Tensor containing the averaged pitch values
                                         for each phoneme. Shape: (batch_size, max_phoneme_len)
    """
    # Initialize placeholder for averaged pitch values, filling with zeros
    pitches_averaged = torch.zeros(
        (pitches.shape[0], max_phoneme_len), device=pitches.device,
    )
    # Loop over each sample in the batch
    for batch_idx in range(durations.shape[0]):
        # Set the starting index of pitch sequence
        start_idx = 0
        # Loop over each phoneme duration
        for i, duration in enumerate(durations[batch_idx]):
            # Convert duration to integer
            duration = duration.int().item()
            # If the duration is not zero
            if duration != 0:
                # Calculate the mean pitch value for the duration of the current phoneme
                mean = torch.mean(pitches[batch_idx, start_idx : start_idx + duration])
                # Store the averaged pitch value
                pitches_averaged[batch_idx][i] = mean
                # Update the starting index for the next phoneme
                start_idx += duration

    # Return tensor with the averaged pitch values
    return pitches_averaged
