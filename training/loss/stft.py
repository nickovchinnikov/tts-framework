import torch


def stft(
    x: torch.Tensor,
    fft_size: int,
    hop_size: int,
    win_length: int,
    window: torch.Tensor,
) -> torch.Tensor:
    r"""Perform STFT and convert to magnitude spectrogram.
    STFT stands for Short-Time Fourier Transform. It is a signal processing technique that is used to analyze the frequency content of a signal over time. The STFT is computed by dividing a long signal into shorter segments, and then computing the Fourier transform of each segment. This results in a time-frequency representation of the signal, where the frequency content of the signal is shown as a function of time.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (torch.Tensor): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    # NOTE (kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)
