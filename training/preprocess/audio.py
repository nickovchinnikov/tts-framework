import sys
from typing import Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio


def stereo_to_mono(audio: torch.FloatTensor) -> torch.FloatTensor:
    r"""
    Converts a stereo audio tensor to mono by taking the mean across channels.

    Args:
        audio (torch.FloatTensor): Input audio tensor of shape (channels, samples).

    Returns:
        torch.FloatTensor: Mono audio tensor of shape (1, samples).
    """
    return torch.mean(audio, axis=0, keepdims=True)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    r"""
    Resamples an audio waveform from the original sampling rate to the target sampling rate.

    Args:
        wav (np.ndarray): The audio waveform to be resampled.
        orig_sr (int): The original sampling rate of the audio waveform.
        target_sr (int): The target sampling rate to resample the audio waveform to.

    Returns:
        np.ndarray: The resampled audio waveform.
    """
    wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    return wav


def safe_load(path: str, sr: Union[int, None]) -> Tuple[np.ndarray, int]:
    r"""
    Load an audio file from disk and return its content as a numpy array.

    Args:
        path (str): The path to the audio file.
        sr (int or None): The target sampling rate. If None, the original sampling rate is used.

    Returns:
        Tuple[np.ndarray, int]: A tuple containing the audio content as a numpy array and the actual sampling rate.
    """
    try:
        audio, sr_actual = torchaudio.load(path)
        if audio.shape[0] > 0:
            audio = stereo_to_mono(audio)
        audio = audio.squeeze(0)
        if sr_actual != sr and sr is not None:
            audio = resample(audio.numpy(), orig_sr=sr_actual, target_sr=sr)
            sr_actual = sr
        else:
            audio = audio.numpy()
    except Exception as e:
        raise type(e)(
            f"The following error happened loading the file {path} ... \n" + str(e)
        ).with_traceback(sys.exc_info()[2])

    return audio, sr_actual


def preprocess_audio(
    audio: torch.FloatTensor, sr_actual: int, sr: Union[int, None]
) -> Tuple[torch.Tensor, int]:
    r"""
    Preprocesses audio by converting stereo to mono, resampling if necessary, and returning the audio tensor and sample rate.

    Args:
        audio (torch.Tensor): The audio tensor to preprocess.
        sr_actual (int): The actual sample rate of the audio.
        sr (Union[int, None]): The target sample rate to resample the audio to, if necessary.

    Returns:
        Tuple[torch.Tensor, int]: The preprocessed audio tensor and sample rate.
    """
    try:
        if audio.shape[0] > 0:
            audio = stereo_to_mono(audio)
        audio = audio.squeeze(0)
        if sr_actual != sr and sr is not None:
            audio = resample(audio.numpy(), orig_sr=sr_actual, target_sr=sr)
            # Convert back to torch tensor
            audio = torch.from_numpy(audio)
            sr_actual = sr
    except Exception as e:
        raise type(e)(
            f"The following error happened while processing the audio ... \n {str(e)}"
        ).with_traceback(sys.exc_info()[2])

    return audio, sr_actual


def normalize_loudness(wav: np.ndarray) -> np.ndarray:
    r"""
    Normalize the loudness of an audio waveform.

    Args:
        wav (np.ndarray): The input waveform.

    Returns:
        np.ndarray: The normalized waveform.

    Examples:
        >>> wav = np.array([1.0, 2.0, 3.0])
        >>> normalize_loudness(wav)
        array([0.33333333, 0.66666667, 1.  ])
    """
    return wav / np.max(np.abs(wav))
