from dataclasses import dataclass
from typing import Any, List, Tuple, Union

from dp.phonemizer import Phonemizer
import numpy as np
from scipy.stats import betabinom
import torch

from model.config import PreprocessingConfig, get_lang_map

from .audio import normalize_loudness, preprocess_audio
from .compute_yin import compute_yin, norm_interp_f0
from .normilize_text import NormilizeText
from .tacotron_stft import TacotronSTFT


@dataclass
class PreprocessAudioResult:
    wav: torch.Tensor
    mel: torch.Tensor
    pitch: torch.Tensor
    phones_ipa: List[str]
    phones: torch.Tensor
    attn_prior: torch.Tensor
    raw_text: str
    normalized_text: str
    speaker_id: int
    chapter_id: int
    utterance_id: str
    pitch_is_normalized: bool


class PreprocessLibriTTS:
    r"""Preprocessing PreprocessLibriTTS audio and text data for use with a TacotronSTFT model.

    Args:
        lang (str): The language of the input text.
        phonemizer_checkpoint (str): The path to the phonemizer checkpoint.

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
        lang: str = "en",
        phonemizer_checkpoint: str = "checkpoints/en_us_cmudict_ipa_forward.pt",
    ):
        super().__init__()

        lang_map = get_lang_map(lang)

        self.phonemizer_lang = lang_map.phonemizer
        normilize_text_lang = lang_map.nemo
        processing_lang_type = lang_map.processing_lang_type

        self.phonemizer = Phonemizer.from_checkpoint(phonemizer_checkpoint)

        self.normilize_text = NormilizeText(normilize_text_lang)

        preprocess_config = PreprocessingConfig(processing_lang_type)

        self.min_seconds = preprocess_config.min_seconds
        self.max_seconds = preprocess_config.max_seconds

        self.hop_length = preprocess_config.stft.hop_length
        self.sampling_rate = preprocess_config.sampling_rate

        self.use_audio_normalization = preprocess_config.use_audio_normalization

        self.filter_length = preprocess_config.stft.filter_length
        self.hop_length = preprocess_config.stft.hop_length
        self.mel_fmin = preprocess_config.stft.mel_fmin

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

    def beta_binomial_prior_distribution(
        self, phoneme_count: int, mel_count: int, scaling_factor: float = 1.0,
    ) -> torch.Tensor:
        r"""Computes the beta-binomial prior distribution for the attention mechanism.

        Args:
            phoneme_count (int): Number of phonemes in the input text.
            mel_count (int): Number of mel frames in the input mel-spectrogram.
            scaling_factor (float, optional): Scaling factor for the beta distribution. Defaults to 1.0.

        Returns:
            torch.Tensor: A 2D tensor containing the prior distribution.
        """
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M + 1):
            a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
            rv: Any = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        result = torch.from_numpy(np.array(mel_text_probs))
        return result

    def __call__(
        self,
        row: Tuple[torch.Tensor, int, str, str, int, int, str],
    ) -> Union[None, PreprocessAudioResult]:
        r"""Preprocesses audio and text data for use with a TacotronSTFT model.

        Args:
            row (Tuple[torch.FloatTensor, int, str, str, int, int, str]): The input row. The row is a tuple containing the following elements: (audio, sr_actual, raw_text, normalized_text, speaker_id, chapter_id, utterance_id).

        Returns:
            dict: A dictionary containing the preprocessed audio and text data.

        Examples:
            >>> preprocess_audio = PreprocessAudio("english_only")
            >>> audio = torch.randn(1, 44100)
            >>> sr_actual = 44100
            >>> raw_text = "Hello, world!"
            >>> output = preprocess_audio(audio, sr_actual, raw_text)
            >>> output.keys()
            dict_keys(['wav', 'mel', 'pitch', 'phones', 'raw_text', 'normalized_text', 'speaker_id', 'chapter_id', 'utterance_id', 'pitch_is_normalized'])
        """
        (
            audio,
            sr_actual,
            raw_text,
            normalized_text,
            speaker_id,
            chapter_id,
            utterance_id,
        ) = row

        wav, sampling_rate = preprocess_audio(audio, sr_actual, self.sampling_rate)

        # TODO: check this, maybe you need to move it to some other place
        # TODO: maybe we can increate the max_samples ?
        if wav.shape[0] < self.min_samples or wav.shape[0] > self.max_samples:
            return None

        if self.use_audio_normalization:
            wav = normalize_loudness(wav)

        normalized_text = self.normilize_text(normalized_text)

        phones_ipa: Any = self.phonemizer(raw_text, lang=self.phonemizer_lang)
        phones = self.phonemizer.predictor.phoneme_tokenizer(
            phones_ipa, language=self.phonemizer_lang,
        )
        # Convert to tensor
        phones = torch.Tensor(phones)

        mel_spectrogram = self.tacotronSTFT.get_mel_from_wav(wav)

        # Skipping small sample due to the mel-spectrogram containing less than self.mel_fmin frames
        if mel_spectrogram.shape[1] < self.mel_fmin:
            return None

        # Text is longer than mel, will be skipped due to monotonic alignment search
        if phones.shape[0] >= mel_spectrogram.shape[1]:
            return None

        pitch, _, _, _ = compute_yin(
            wav,
            sr=sampling_rate,
            w_len=self.filter_length,
            w_step=self.hop_length,
            f0_min=50,
            f0_max=1000,
            harmo_thresh=0.25,
        )

        # Skipping pitch that sum less or equal to 1
        if np.sum(pitch != 0) <= 1:
            return None

        pitch, _ = norm_interp_f0(pitch)

        pitch = torch.from_numpy(pitch)

        # TODO this shouldnt be necessary, currently pitch sometimes has 1 less frame than spectrogram,
        # We should find out why
        mel_spectrogram = mel_spectrogram[:, : pitch.shape[0]]

        attn_prior = self.beta_binomial_prior_distribution(
            phones.shape[0], mel_spectrogram.shape[1],
        ).T

        assert pitch.shape[0] == mel_spectrogram.shape[1], (
            pitch.shape,
            mel_spectrogram.shape[1],
        )

        return PreprocessAudioResult(
            wav=wav,
            mel=mel_spectrogram,
            pitch=pitch,
            attn_prior=attn_prior,
            phones_ipa=phones_ipa,
            phones=phones,
            raw_text=raw_text,
            normalized_text=normalized_text,
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            utterance_id=utterance_id,
            # TODO: check the pitch normalization process
            pitch_is_normalized=False,
        )
