from dataclasses import dataclass

import torch
from torch import nn

import matplotlib.pyplot as plt
import numpy as np
import librosa

from torchmetrics.audio import (
    ComplexScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    ShortTimeObjectiveIntelligibility,
    PerceptualEvaluationSpeechQuality,
)

from model.config import PreprocessingConfig, get_lang_map
from training.preprocess.audio_processor import AudioProcessor


@dataclass
class MetricsResult:
    r"""
    A data class that holds the results of the computed metrics.

    Attributes:
        energy (torch.Tensor): The energy loss ratio.
        si_sdr (torch.Tensor): The scale-invariant signal-to-distortion ratio.
        si_snr (torch.Tensor): The scale-invariant signal-to-noise ratio.
        c_si_snr (torch.Tensor): The complex scale-invariant signal-to-noise ratio.
        stoi (torch.Tensor): The short-time objective intelligibility.
        pesq (torch.Tensor): The perceptual evaluation of speech quality.
    """
    energy: torch.Tensor
    si_sdr: torch.Tensor
    si_snr: torch.Tensor
    c_si_snr: torch.Tensor
    stoi: torch.Tensor
    pesq: torch.Tensor


class Metrics:
    r"""
    A class that computes various audio metrics.

    Attributes:
        hop_length (int): The hop length for the STFT.
        filter_length (int): The filter length for the STFT.
        mel_fmin (int): The minimum frequency for the Mel scale.
        win_length (int): The window length for the STFT.
        audio_processor (AudioProcessor): The audio processor.
        mse_loss (nn.MSELoss): The mean squared error loss.
        si_sdr (ScaleInvariantSignalDistortionRatio): The scale-invariant signal-to-distortion ratio.
        si_snr (ScaleInvariantSignalNoiseRatio): The scale-invariant signal-to-noise ratio.
        c_si_snr (ComplexScaleInvariantSignalNoiseRatio): The complex scale-invariant signal-to-noise ratio.
        stoi (ShortTimeObjectiveIntelligibility): The short-time objective intelligibility.
        pesq (PerceptualEvaluationSpeechQuality): The perceptual evaluation of speech quality.
    """
    def __init__(self, lang: str = "en"):
        lang_map = get_lang_map(lang)
        preprocess_config = PreprocessingConfig(lang_map.processing_lang_type)

        self.hop_length = preprocess_config.stft.hop_length
        self.filter_length = preprocess_config.stft.filter_length
        self.mel_fmin = preprocess_config.stft.mel_fmin
        self.win_length = preprocess_config.stft.win_length

        self.audio_processor = AudioProcessor()
        self.mse_loss = nn.MSELoss()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.c_si_snr = ComplexScaleInvariantSignalNoiseRatio(zero_mean=False)
        self.stoi = ShortTimeObjectiveIntelligibility(1000, extended=True)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode="wb")
    
    def __call__(
            self,
            wav_predictions: torch.Tensor,
            wav_targets: torch.Tensor,
            mel_predictions: torch.Tensor,
            mel_targets: torch.Tensor
    ) -> MetricsResult:
        r"""
        Compute the metrics.

        Args:
            wav_predictions (torch.Tensor): The predicted waveforms.
            wav_targets (torch.Tensor): The target waveforms.
            mel_predictions (torch.Tensor): The predicted Mel spectrograms.
            mel_targets (torch.Tensor): The target Mel spectrograms.

        Returns:
            MetricsResult: The computed metrics.
        """
        wav_predictions_energy = self.audio_processor.wav_to_energy(
            wav_predictions.unsqueeze(0),
            self.filter_length,
            self.hop_length,
            self.win_length
        )

        wav_targets_energy = self.audio_processor.wav_to_energy(
            wav_targets.unsqueeze(0),
            self.filter_length,
            self.hop_length,
            self.win_length
        )
        
        energy: torch.Tensor = self.mse_loss(wav_predictions_energy, wav_targets_energy)

        self.si_sdr.to(wav_predictions.device)
        self.si_snr.to(wav_predictions.device)
        self.c_si_snr.to(wav_predictions.device)

        # New Metrics
        si_sdr: torch.Tensor = self.si_sdr(mel_predictions, mel_targets)
        si_snr: torch.Tensor = self.si_snr(mel_predictions, mel_targets)

        # New shape: [1, F, T, 2]
        mel_predictions_complex = torch.stack((mel_predictions, torch.zeros_like(mel_predictions)), dim=-1)  
        mel_targets_complex = torch.stack((mel_targets, torch.zeros_like(mel_targets)), dim=-1)
        c_si_snr: torch.Tensor = self.c_si_snr(mel_predictions_complex, mel_targets_complex)
        
        stoi: torch.Tensor = self.stoi(mel_predictions, mel_targets)
        pesq: torch.Tensor = self.pesq(wav_predictions, wav_targets)

        return MetricsResult(
            energy,
            si_sdr,
            si_snr,
            c_si_snr,
            stoi,
            pesq,
        )

    def plot_spectrograms(self, mel_target: np.ndarray, mel_prediction: np.ndarray, sr: int = 22050):
        r"""
        Plots the mel spectrograms for the target and the prediction.
        """
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, dpi=80)

        img1 = librosa.display.specshow(mel_target, x_axis='time', y_axis='mel', sr=sr, ax=axs[0])
        axs[0].set_title('Target spectrogram')
        fig.colorbar(img1, ax=axs[0], format='%+2.0f dB')

        img2 = librosa.display.specshow(mel_prediction, x_axis='time', y_axis='mel', sr=sr, ax=axs[1])
        axs[1].set_title('Prediction spectrogram')
        fig.colorbar(img2, ax=axs[1], format='%+2.0f dB')

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.5)

        return fig

    def plot_spectrograms_fast(self, mel_target: np.ndarray, mel_prediction: np.ndarray, sr: int = 22050):
        r"""
        Plots the mel spectrograms for the target and the prediction.
        """
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

        axs[0].specgram(
            mel_target,
            aspect='auto',
            Fs=sr,
            cmap=plt.get_cmap("magma") # type: ignore
        )
        axs[0].set_title('Target spectrogram')

        axs[1].specgram(
            mel_prediction,
            aspect='auto',
            Fs=sr,
            cmap=plt.get_cmap("magma") # type: ignore
        )
        axs[1].set_title('Prediction spectrogram')

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.5)

        return fig
