from typing import Literal, Optional, Tuple, Union

import librosa
import torch
from torch import Tensor, nn
from torch.nn import functional


class TorchSTFT(nn.Module):
    r"""Some of the audio processing funtions using Torch for faster batch processing.

    Args:
        n_fft (int): FFT window size for STFT.
        hop_length (int): number of frames between STFT columns.
        win_length (int, optional): STFT window length.
        pad_wav (bool, optional): If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.
        window (str, optional): The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"
        sample_rate (int, optional): target audio sampling rate. Defaults to None.
        mel_fmin (int, optional): minimum filter frequency for computing melspectrograms. Defaults to None.
        mel_fmax (int, optional): maximum filter frequency for computing melspectrograms. Defaults to None.
        n_mels (int, optional): number of melspectrogram dimensions. Defaults to None.
        use_mel (bool, optional): If True compute the melspectrograms otherwise. Defaults to False.
        do_amp_to_db_linear (bool, optional): enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.
        spec_gain (float, optional): gain applied when converting amplitude to DB. Defaults to 1.0.
        power (float, optional): Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.
        use_htk (bool, optional): Use HTK formula in mel filter instead of Slaney.
        mel_norm (None, 'slaney', or number, optional): If 'slaney', divide the triangular mel weights by the width of the mel band (area normalization).

        If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
        See `librosa.util.normalize` for a full description of supported norm values (including `+-np.inf`).
        Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: int,
        pad_wav: bool = False,
        window: str = "hann_window",
        sample_rate: int = 22050,
        mel_fmin: int = 0,
        mel_fmax: Optional[int] = None,
        n_mels: int = 80,
        use_mel: bool = False,
        do_amp_to_db:bool = False,
        spec_gain: float = 1.0,
        power: Optional[float] = None,
        use_htk: bool = False,
        mel_norm: Optional[Union[Literal["slaney"], float]] = "slaney",
        normalized: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.normalized = normalized

        self.mel_norm: Optional[Union[Literal["slaney"], float]] = mel_norm
        self.mel_basis = None

        if use_mel:
            self._build_mel_basis()

    def __call__(self, x: Tensor):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")

        # B x D x T x 2
        o = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )

        M = o[:, :, :, 0]
        P = o[:, :, :, 1]

        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel and self.mel_basis is not None:
            S = torch.matmul(self.mel_basis.to(x), S)

        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)

        return S

    def _build_mel_basis(self):
        r"""Builds the mel basis for the spectrogram transformation.
        This method is called during initialization if use_mel is set to True.
        """
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    @staticmethod
    def _amp_to_db(x: Tensor, spec_gain: float = 1.0) -> Tensor:
        r"""Converts amplitude to decibels.

        Args:
        x (Tensor): The amplitude tensor to convert.
        spec_gain (float, optional): The gain applied when converting. Defaults to 1.0.

        Returns:
        Tensor: The converted tensor in decibels.
        """
        return torch.log(torch.clamp(x, min=1e-5) * spec_gain)

    @staticmethod
    def _db_to_amp(x: Tensor, spec_gain: float = 1.0) -> Tensor:
        r"""Converts decibels to amplitude.

        Args:
        x (Tensor): The decibel tensor to convert.
        spec_gain (float, optional): The gain applied when converting. Defaults to 1.0.

        Returns:
        Tensor: The converted tensor in amplitude.
        """
        return torch.exp(x) / spec_gain


class STFTLoss(nn.Module):
    r"""STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf

    Attributes:
        n_fft (int): The FFT size.
        hop_length (int): The hop (stride) size.
        win_length (int): The window size.
        stft (TorchSTFT): The STFT function.

    Methods:
        forward(y_hat: Tensor, y: Tensor)
            Compute the STFT loss.
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        r"""Constructs all the necessary attributes for the STFTLoss object.

        Args:
            n_fft (int): The FFT size.
            hop_length (int): The hop (stride) size.
            win_length (int): The window size.
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)

    def forward(self, y_hat: Tensor, y: Tensor):
        r"""Compute the STFT loss.

        Args:
            y_hat (Tensor): The generated waveforms.
            y (Tensor): The real waveforms.

        Returns:
            loss_mag (Tensor): The magnitude loss.
            loss_sc (Tensor): The spectral convergence loss.
        """
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)

        # magnitude loss
        loss_mag = functional.l1_loss(torch.log(y_M), torch.log(y_hat_M))

        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf

    Attributes:
        loss_funcs (torch.nn.ModuleList): A list of STFTLoss modules for different scales.

    Methods:
        forward(y_hat: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]
            Compute the multi-scale STFT loss.
    """

    def __init__(
            self,
            n_ffts: Tuple[int, int, int] = (1024, 2048, 512),
            hop_lengths: Tuple[int, int, int] = (120, 240, 50),
            win_lengths: Tuple[int, int, int] = (600, 1200, 240),
    ):
        r"""Initialize the MultiScaleSTFTLoss module.

        Args:
            n_ffts (Tuple[int, int, int], optional): The FFT sizes for the STFTLoss modules. Defaults to (1024, 2048, 512).
            hop_lengths (Tuple[int, int, int], optional): The hop lengths for the STFTLoss modules. Defaults to (120, 240, 50).
            win_lengths (Tuple[int, int, int], optional): The window lengths for the STFTLoss modules. Defaults to (600, 1200, 240).
        """
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat: Tensor, y: Tensor):
        r"""Compute the multi-scale STFT loss.

        Args:
            y_hat (Tensor): The generated waveforms.
            y (Tensor): The real waveforms.

        Returns:
            Tuple[Tensor, Tensor]: The magnitude and spectral convergence losses.
        """
        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc
