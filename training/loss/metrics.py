from dataclasses import dataclass

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torchaudio.transforms as T
from torchmetrics.audio import (
    ComplexScaleInvariantSignalNoiseRatio,
    ScaleInvariantSignalDistortionRatio,
    ScaleInvariantSignalNoiseRatio,
    SpeechReverberationModulationEnergyRatio,
)

from models.config import PreprocessingConfig, get_lang_map
from training.preprocess.audio_processor import AudioProcessor


@dataclass
class MetricsResult:
    r"""A data class that holds the results of the computed metrics.

    Attributes:
        energy (torch.Tensor): The energy loss ratio.
        si_sdr (torch.Tensor): The scale-invariant signal-to-distortion ratio.
        si_snr (torch.Tensor): The scale-invariant signal-to-noise ratio.
        c_si_snr (torch.Tensor): The complex scale-invariant signal-to-noise ratio.
        mcd (torch.Tensor): The Mel cepstral distortion.
        spec_dist (torch.Tensor): The spectrogram distance.
        f0_rmse (float): The F0 RMSE.
        jitter (float): The jitter.
        shimmer (float): The shimmer.
    """

    energy: torch.Tensor
    si_sdr: torch.Tensor
    si_snr: torch.Tensor
    c_si_snr: torch.Tensor
    mcd: torch.Tensor
    spec_dist: torch.Tensor
    f0_rmse: float
    jitter: float
    shimmer: float


class Metrics:
    r"""A class that computes various audio metrics.

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
    """

    def __init__(
        self,
        lang: str = "en",
    ):
        lang_map = get_lang_map(lang)
        preprocess_config = PreprocessingConfig(lang_map.processing_lang_type)

        self.hop_length = preprocess_config.stft.hop_length
        self.filter_length = preprocess_config.stft.filter_length
        self.mel_fmin = preprocess_config.stft.mel_fmin
        self.win_length = preprocess_config.stft.win_length
        self.sample_rate = preprocess_config.sampling_rate

        self.audio_processor = AudioProcessor()
        self.mse_loss = nn.MSELoss()
        self.si_sdr = ScaleInvariantSignalDistortionRatio()
        self.si_snr = ScaleInvariantSignalNoiseRatio()
        self.c_si_snr = ComplexScaleInvariantSignalNoiseRatio(zero_mean=False)
        self.reverb_modulation_energy_ratio = SpeechReverberationModulationEnergyRatio(self.sample_rate)

    def calculate_mcd(
        self,
        wav_targets: torch.Tensor,
        wav_predictions: torch.Tensor,
        n_mfcc: int = 13,
    ) -> torch.Tensor:
        """Calculate Mel Cepstral Distortion."""
        mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": 400,
                "hop_length": 160,
                "n_mels": 23,
                "center": False,
            },
        ).to(wav_targets.device)
        wav_predictions = wav_predictions.to(wav_targets.device)

        ref_mfcc = mfcc_transform(wav_targets)
        synth_mfcc = mfcc_transform(wav_predictions)

        mcd = torch.mean(torch.sqrt(
            torch.sum((ref_mfcc - synth_mfcc) ** 2, dim=0),
        ))

        return mcd

    def calculate_spectrogram_distance(
        self,
        wav_targets: torch.Tensor,
        wav_predictions: torch.Tensor,
        n_fft: int = 2048,
        hop_length: int = 512,
    ) -> torch.Tensor:
        """Calculate spectrogram distance."""
        spec_transform = T.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            power=None,
        ).to(wav_targets.device)
        wav_predictions = wav_predictions.to(wav_targets.device)

        # Compute the spectrograms
        S1 = spec_transform(wav_targets)
        S2 = spec_transform(wav_predictions)

        # Compute the magnitude spectrograms
        S1_mag = torch.abs(S1)
        S2_mag = torch.abs(S2)

        # Compute the Euclidean distance
        dist = torch.dist(S1_mag.flatten(), S2_mag.flatten())

        return dist

    def calculate_f0_rmse(
        self,
        wav_targets: torch.Tensor,
        wav_predictions: torch.Tensor,
        frame_length: int = 2048,
        hop_length: int = 512,
    ) -> float:
        """Calculate F0 RMSE."""
        wav_targets_ = wav_targets.detach().cpu().numpy()
        wav_predictions_ = wav_predictions.detach().cpu().numpy()

        # Compute the F0 contour for each audio signal
        f0_audio1 = torch.from_numpy(
            librosa.yin(
                wav_targets_,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=self.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            ),
        )
        f0_audio2 = torch.from_numpy(
            librosa.yin(
                wav_predictions_,
                fmin=float(librosa.note_to_hz("C2")),
                fmax=float(librosa.note_to_hz("C7")),
                sr=self.sample_rate,
                frame_length=frame_length,
                hop_length=hop_length,
            ),
        )

        # Assuming f0_audio1 and f0_audio2 are PyTorch tensors
        rmse = torch.sqrt(torch.mean((f0_audio1 - f0_audio2)**2)).item()

        return rmse

    def calculate_jitter_shimmer(
        self,
        audio: torch.Tensor,
    ) -> tuple[float, float]:
        r"""Calculate jitter and shimmer of an audio signal.

        Jitter and shimmer are two metrics used in speech signal processing to measure the quality of voice signals.

        Jitter refers to the short-term variability of a signal's fundamental frequency (F0). It is often used as an indicator of voice disorders, as high levels of jitter can indicate a lack of control over the vocal folds.

        Shimmer, on the other hand, refers to the short-term variability in amplitude of the voice signal. Like jitter, high levels of shimmer can be indicative of voice disorders, as they can suggest a lack of control over the vocal tract.

        Summary:
        Jitter is the short-term variability of a signal's fundamental frequency (F0).
        Shimmer is the short-term variability in amplitude of the voice signal.

        Args:
            audio (torch.Tensor): The audio signal to analyze.

        Returns:
            tuple[float, float]: The calculated jitter and shimmer values.
        """
        # Create a transformation to calculate the spectrogram
        spectrogram = T.Spectrogram(
            n_fft=self.filter_length * 2,
            hop_length=self.hop_length * 2,
            power=None,
        )

        # Calculate the spectrogram of the audio signal
        amplitude = spectrogram(audio)

        # Calculate the F0 contour using the yin method
        f0 = T.Vad(sample_rate=self.sample_rate)(audio)

        # Calculate the relative changes in the F0 and amplitude contours
        jitter = torch.mean(
            torch.abs(torch.diff(f0, dim=-1)) / torch.diff(f0, dim=-1),
        ).item()
        shimmer = torch.mean(
            torch.abs(torch.diff(amplitude, dim=-1)) / torch.diff(amplitude, dim=-1),
        )

        shimmer = torch.abs(shimmer).item()

        return jitter, shimmer

    def wav_metrics(self, wav_predictions: torch.Tensor):
        r"""Compute the metrics for the waveforms.

        Args:
            wav_predictions (torch.Tensor): The predicted waveforms.

        Returns:
            tuple[float, float, float]: The computed metrics.
        """
        ermr = self.reverb_modulation_energy_ratio(wav_predictions).item()
        jitter, shimmer = self.calculate_jitter_shimmer(wav_predictions)

        return (
            ermr,
            jitter,
            shimmer,
        )

    def __call__(
        self,
        wav_predictions: torch.Tensor,
        wav_targets: torch.Tensor,
        mel_predictions: torch.Tensor,
        mel_targets: torch.Tensor,
    ) -> MetricsResult:
        r"""Compute the metrics.

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
            self.win_length,
        )

        wav_targets_energy = self.audio_processor.wav_to_energy(
            wav_targets.unsqueeze(0),
            self.filter_length,
            self.hop_length,
            self.win_length,
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

        mcd = self.calculate_mcd(wav_targets, wav_predictions)
        spec_dist = self.calculate_spectrogram_distance(wav_targets, wav_predictions)
        f0_rmse = self.calculate_f0_rmse(wav_targets, wav_predictions)
        jitter, shimmer = self.calculate_jitter_shimmer(wav_predictions)

        return MetricsResult(
            energy,
            si_sdr,
            si_snr,
            c_si_snr,
            mcd,
            spec_dist,
            f0_rmse,
            jitter,
            shimmer,
        )

    def plot_spectrograms(self, mel_target: np.ndarray, mel_prediction: np.ndarray, sr: int = 22050):
        r"""Plots the mel spectrograms for the target and the prediction."""
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, dpi=80)

        img1 = librosa.display.specshow(mel_target, x_axis="time", y_axis="mel", sr=sr, ax=axs[0])
        axs[0].set_title("Target spectrogram")
        fig.colorbar(img1, ax=axs[0], format="%+2.0f dB")

        img2 = librosa.display.specshow(mel_prediction, x_axis="time", y_axis="mel", sr=sr, ax=axs[1])
        axs[1].set_title("Prediction spectrogram")
        fig.colorbar(img2, ax=axs[1], format="%+2.0f dB")

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.5)

        return fig

    def plot_spectrograms_fast(
        self,
        mel_target: np.ndarray,
        mel_prediction: np.ndarray,
        sr: int = 22050,
    ):
        r"""Plots the mel spectrograms for the target and the prediction."""
        fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)

        axs[0].specgram(
            mel_target,
            aspect="auto",
            Fs=sr,
            cmap=plt.get_cmap("magma"), # type: ignore
        )
        axs[0].set_title("Target spectrogram")

        axs[1].specgram(
            mel_prediction,
            aspect="auto",
            Fs=sr,
            cmap=plt.get_cmap("magma"), # type: ignore
        )
        axs[1].set_title("Prediction spectrogram")

        # Adjust the spacing between subplots
        fig.subplots_adjust(hspace=0.5)

        return fig
