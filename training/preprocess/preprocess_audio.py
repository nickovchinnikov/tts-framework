from dataclasses import dataclass
from typing import Union

import lightning.pytorch as pl
import numpy as np
import torch

from model.config import PreprocessingConfig

from .audio import normalize_loudness, preprocess_audio
from .compute_yin import compute_yin, norm_interp_f0
from .tacotron_stft import TacotronSTFT
from .text import byte_encode


@dataclass
class PreprocessAudioResult:
    waw: torch.FloatTensor
    mel: torch.FloatTensor
    pitch: torch.FloatTensor
    phones: bytes
    raw_text: str
    pitch_is_normalized: bool


class PreprocessAudio(pl.LightningModule):
    r"""
    A PyTorch Lightning module for preprocessing audio and text data for use with a TacotronSTFT model.

    Args:
        preprocess_config (PreprocessingConfig): The preprocessing configuration.

    Attributes:
        min_seconds (float): The minimum duration of audio clips in seconds.
        max_seconds (float): The maximum duration of audio clips in seconds.
        hop_length (int): The hop length of the STFT.
        sampling_rate (int): The sampling rate of the audio.
        use_audio_normalization (bool): Whether to normalize the loudness of the audio.
        tacotronSTFT (TacotronSTFT): The TacotronSTFT object used for computing mel spectrograms.
        min_samples (int): The minimum number of audio samples in a clip.
        max_samples (int): The maximum number of audio samples in a clip.
    """

    def __init__(
        self,
        preprocess_config: PreprocessingConfig,
    ):
        super().__init__()

        self.min_seconds = preprocess_config.min_seconds
        self.max_seconds = preprocess_config.max_seconds

        self.hop_length = preprocess_config.stft.hop_length
        self.sampling_rate = preprocess_config.sampling_rate

        self.use_audio_normalization = preprocess_config.use_audio_normalization

        self.filter_length = preprocess_config.stft.filter_length
        self.hop_length = preprocess_config.stft.hop_length

        self.tacotronSTFT = TacotronSTFT(
            filter_length=preprocess_config.stft.filter_length,
            hop_length=preprocess_config.stft.hop_length,
            win_length=preprocess_config.stft.win_length,
            n_mel_channels=preprocess_config.stft.n_mel_channels,
            sampling_rate=self.sampling_rate,
            mel_fmin=preprocess_config.stft.mel_fmin,
            mel_fmax=preprocess_config.stft.mel_fmax,
            center=False,
        )

        self.min_samples = int(self.sampling_rate * self.min_seconds)
        self.max_samples = int(self.sampling_rate * self.max_seconds)

    def forward(
        self, audio: torch.FloatTensor, sr_actual: int, raw_text: str
    ) -> Union[None, PreprocessAudioResult]:
        r"""
        Preprocesses audio and text data for use with a TacotronSTFT model.

        Args:
            audio (torch.FloatTensor): The input audio waveform.
            sr_actual (int): The actual sampling rate of the input audio.
            raw_text (str): The raw input text.

        Returns:
            dict: A dictionary containing the preprocessed audio and text data.

        Examples:
            >>> preprocess_config = PreprocessingConfig()
            >>> preprocess_audio = PreprocessAudio(preprocess_config)
            >>> audio = torch.randn(1, 44100)
            >>> sr_actual = 44100
            >>> raw_text = "Hello, world!"
            >>> output = preprocess_audio(audio, sr_actual, raw_text)
            >>> output.keys()
            dict_keys(['wav', 'mel', 'pitch', 'phones', 'raw_text', 'pitch_is_normalized'])
        """
        wav, sampling_rate = preprocess_audio(audio, sr_actual, self.sampling_rate)

        # TODO: check this, maybe you need to move it to some other place
        if wav.shape[0] < self.min_samples or wav.shape[0] > self.max_samples:
            return

        if self.use_audio_normalization:
            wav = normalize_loudness(wav)

        phones = byte_encode(raw_text.strip())

        mel_spectrogram = self.tacotronSTFT.get_mel_from_wav(wav)

        pitch, _, _, _ = compute_yin(
            wav,
            sr=sampling_rate,
            w_len=self.filter_length,
            w_step=self.hop_length,
            f0_min=50,
            f0_max=1000,
            harmo_thresh=0.25,
        )

        # TODO: check this, maybe you need to move it to some other place
        if np.sum(pitch != 0) <= 1:
            return

        # TODO this shouldnt be necessary, currently pitch sometimes has 1 less frame than spectrogram,
        # We should find out why
        mel_spectrogram = mel_spectrogram[:, : pitch.shape[0]]

        pitch, _ = norm_interp_f0(pitch)

        assert pitch.shape[0] == mel_spectrogram.shape[1], (
            pitch.shape,
            mel_spectrogram.shape[1],
        )

        return {
            "wav": wav,
            "mel": mel_spectrogram,
            "pitch": torch.from_numpy(pitch),
            "phones": phones,
            "raw_text": raw_text,
            # TODO: check the pitch normalization process
            "pitch_is_normalized": False,
        }
