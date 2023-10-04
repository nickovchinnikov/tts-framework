from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

######################################################################
# Original implementation from https://github.com/NVIDIA/mellotron/blob/master/yin.py
######################################################################


def differenceFunction(x: np.ndarray, N: int, tau_max: int) -> np.ndarray:
    r"""
    Compute the difference function of an audio signal.

    This function computes the difference function of an audio signal `x` using the algorithm described in equation (6) of [1]. The difference function is a measure of the similarity between the signal and a time-shifted version of itself, and is commonly used in pitch detection algorithms.

    This implementation uses the NumPy FFT functions to compute the difference function efficiently.

    Parameters:
        x (np.ndarray): The audio signal to compute the difference function for.
        N (int): The length of the audio signal.
        tau_max (int): The maximum integration window size to use.

    Returns:
        np.ndarray: The difference function of the audio signal.

    References:
        [1] A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music," The Journal of the Acoustical Society of America, vol. 111, no. 4, pp. 1917-1930, 2002.
    """
    x = np.array(x, np.float64)
    w = x.size
    tau_max = min(tau_max, w)
    x_cumsum = np.concatenate((np.array([0.0]), (x * x).cumsum()))
    size = w + tau_max
    p2 = (size // 32).bit_length()
    nice_numbers = (16, 18, 20, 24, 25, 27, 30, 32)
    size_pad = min(x * 2**p2 for x in nice_numbers if x * 2**p2 >= size)
    fc = np.fft.rfft(x, size_pad)
    conv = np.fft.irfft(fc * fc.conjugate())[:tau_max]
    return x_cumsum[w : w - tau_max : -1] + x_cumsum[w] - x_cumsum[:tau_max] - 2 * conv


def cumulativeMeanNormalizedDifferenceFunction(df: np.ndarray, N: int) -> np.ndarray:
    r"""
    Compute the cumulative mean normalized difference function (CMND) of a difference function.

    The CMND is defined as the element-wise product of the difference function with a range of values from 1 to N-1,
    divided by the cumulative sum of the difference function up to that point, plus a small epsilon value to avoid
    division by zero. The first element of the CMND is set to 1.

    Args:
        df (np.ndarray): The difference function.
        N (int): The length of the data.

    Returns:
        np.ndarray: The cumulative mean normalized difference function.

    References:
        [1] K. K. Paliwal and R. P. Sharma, "A robust algorithm for pitch detection in noisy speech signals,"
            Speech Communication, vol. 12, no. 3, pp. 249-263, 1993.
    """
    cmndf = (
        df[1:] * range(1, N) / (np.cumsum(df[1:]).astype(float) + 1e-8)
    )  # scipy method
    return np.insert(cmndf, 0, 1)


def getPitch(cmdf: np.ndarray, tau_min: int, tau_max: int, harmo_th=0.1) -> int:
    r"""
    Compute the fundamental period of a frame based on the Cumulative Mean Normalized Difference function (CMND).

    The CMND is a measure of the periodicity of a signal, and is computed as the cumulative mean normalized difference
    function of the difference function of the signal. The fundamental period is the first value of the index `tau`
    between `tau_min` and `tau_max` where the CMND is below the `harmo_th` threshold. If there are no such values, the
    function returns 0 to indicate that the signal is unvoiced.

    Args:
        cmdf (np.ndarray): The Cumulative Mean Normalized Difference function of the signal.
        tau_min (int): The minimum period for speech.
        tau_max (int): The maximum period for speech.
        harmo_th (float, optional): The harmonicity threshold to determine if it is necessary to compute pitch
            frequency. Defaults to 0.1.

    Returns:
        int: The fundamental period of the signal, or 0 if the signal is unvoiced.

    References:
        [1] K. K. Paliwal and R. P. Sharma, "A robust algorithm for pitch detection in noisy speech signals,"
            Speech Communication, vol. 12, no. 3, pp. 249-263, 1993.
    """
    tau = tau_min
    while tau < tau_max:
        if cmdf[tau] < harmo_th:
            while tau + 1 < tau_max and cmdf[tau + 1] < cmdf[tau]:
                tau += 1
            return tau
        tau += 1

    return 0  # if unvoiced


def compute_yin(
    sig: np.ndarray,
    sr: int,
    w_len: int = 512,
    w_step: int = 256,
    f0_min: int = 100,
    f0_max: int = 500,
    harmo_thresh: float = 0.1,
) -> Tuple[np.ndarray, List[float], List[float], List[float]]:
    r"""
    Compute the Yin Algorithm for pitch detection on an audio signal.

    The Yin Algorithm is a widely used method for pitch detection in speech and music signals. It works by computing the
    Cumulative Mean Normalized Difference function (CMND) of the difference function of the signal, and finding the first
    minimum of the CMND below a given threshold. The fundamental period of the signal is then estimated as the inverse of
    the lag corresponding to this minimum.

    Args:
        sig (np.ndarray): The audio signal as a 1D numpy array of floats.
        sr (int): The sampling rate of the signal.
        w_len (int, optional): The size of the analysis window in samples. Defaults to 512.
        w_step (int, optional): The size of the lag between two consecutive windows in samples. Defaults to 256.
        f0_min (int, optional): The minimum fundamental frequency that can be detected in Hz. Defaults to 100.
        f0_max (int, optional): The maximum fundamental frequency that can be detected in Hz. Defaults to 500.
        harmo_thresh (float, optional): The threshold of detection. The algorithm returns the first minimum of the CMND
            function below this threshold. Defaults to 0.1.

    Returns:
        Tuple[np.ndarray, List[float], List[float], List[float]]: A tuple containing the following elements:
            * pitches (np.ndarray): A 1D numpy array of fundamental frequencies estimated for each analysis window.
            * harmonic_rates (List[float]): A list of harmonic rate values for each fundamental frequency value, which
              can be interpreted as a confidence value.
            * argmins (List[float]): A list of the minimums of the Cumulative Mean Normalized Difference Function for
              each analysis window.
            * times (List[float]): A list of the time of each estimation, in seconds.

    References:
        [1] A. K. Jain, Fundamentals of Digital Image Processing, Prentice Hall, 1989.
        [2] A. de Cheveigné and H. Kawahara, "YIN, a fundamental frequency estimator for speech and music," The Journal
            of the Acoustical Society of America, vol. 111, no. 4, pp. 1917-1930, 2002.
    """
    sig_torch = torch.from_numpy(sig)
    sig_torch = sig_torch.view(1, 1, -1)
    sig_torch = F.pad(
        sig_torch.unsqueeze(1),
        (int((w_len - w_step) / 2), int((w_len - w_step) / 2), 0, 0),
        mode="reflect",
    )
    sig_torch = sig_torch.view(-1).numpy()

    tau_min = int(sr / f0_max)
    tau_max = int(sr / f0_min)

    timeScale = range(
        0, len(sig_torch) - w_len, w_step
    )  # time values for each analysis window
    times = [t / float(sr) for t in timeScale]
    frames = [sig_torch[t : t + w_len] for t in timeScale]

    pitches = [0.0] * len(timeScale)
    harmonic_rates = [0.0] * len(timeScale)
    argmins = [0.0] * len(timeScale)

    for i, frame in enumerate(frames):
        # Compute YIN
        df = differenceFunction(frame, w_len, tau_max)
        cmdf = cumulativeMeanNormalizedDifferenceFunction(df, tau_max)
        p = getPitch(cmdf, tau_min, tau_max, harmo_thresh)

        # Get results
        if np.argmin(cmdf) > tau_min:
            argmins[i] = float(sr / np.argmin(cmdf))
        if p != 0:  # A pitch was found
            pitches[i] = float(sr / p)
            harmonic_rates[i] = cmdf[p]
        else:  # No pitch, but we compute a value of the harmonic rate
            harmonic_rates[i] = min(cmdf)

    return np.array(pitches), harmonic_rates, argmins, times
