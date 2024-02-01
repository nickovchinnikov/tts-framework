import torch

from librosa.filters import mel as librosa_mel_fn


class AudioProcessor:
    r"""
    A class used to process audio signals and convert them into different representations.

    Attributes:
        hann_window (dict): A dictionary to store the Hann window for different configurations.
        mel_basis (dict): A dictionary to store the Mel basis for different configurations.

    Methods:
        name_mel_basis(spec, n_fft, fmax): Generate a name for the Mel basis based on the FFT size, maximum frequency, data type, and device.
        amp_to_db(magnitudes, C=1, clip_val=1e-5): Convert amplitude to decibels (dB).
        db_to_amp(magnitudes, C=1): Convert decibels (dB) to amplitude.
        wav_to_spec(y, n_fft, hop_length, win_length, center=False): Convert a waveform to a spectrogram and compute the magnitude.
        wav_to_energy(y, n_fft, hop_length, win_length, center=False): Convert a waveform to a spectrogram and compute the energy.
        spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax): Convert a spectrogram to a Mel spectrogram.
        wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False): Convert a waveform to a Mel spectrogram.
    """
    def __init__(self):
        self.hann_window = {}
        self.mel_basis = {}

    @staticmethod
    def name_mel_basis(spec: torch.Tensor, n_fft: int, fmax: int) -> str:
        """
        Generate a name for the Mel basis based on the FFT size, maximum frequency, data type, and device.

        Args:
            spec (torch.Tensor): The spectrogram tensor.
            n_fft (int): The FFT size.
            fmax (int): The maximum frequency.

        Returns:
            str: The generated name for the Mel basis.
        """
        n_fft_len = f"{n_fft}_{fmax}_{spec.dtype}_{spec.device}"
        return n_fft_len

    @staticmethod
    def amp_to_db(magnitudes: torch.Tensor, C: int = 1, clip_val: float = 1e-5) -> torch.Tensor:
        r"""
        Convert amplitude to decibels (dB).

        Args:
            magnitudes (Tensor): The amplitude magnitudes to convert.
            C (int, optional): A constant value used in the conversion. Defaults to 1.
            clip_val (float, optional): A value to clamp the magnitudes to avoid taking the log of zero. Defaults to 1e-5.

        Returns:
            Tensor: The converted magnitudes in dB.
        """

        return torch.log(torch.clamp(magnitudes, min=clip_val) * C)

    @staticmethod
    def db_to_amp(magnitudes: torch.Tensor, C: int = 1) -> torch.Tensor:
        r"""
        Convert decibels (dB) to amplitude.

        Args:
            magnitudes (Tensor): The dB magnitudes to convert.
            C (int, optional): A constant value used in the conversion. Defaults to 1.

        Returns:
            Tensor: The converted magnitudes in amplitude.
        """

        return torch.exp(magnitudes) / C

    def wav_to_spec(
        self,
        y: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool = False
    ) -> torch.Tensor:
        r"""
        Convert a waveform to a spectrogram and compute the magnitude.

        Args:
            y (Tensor): The input waveform.
            n_fft (int): The FFT size.
            hop_length (int): The hop (stride) size.
            win_length (int): The window size.
            center (bool, optional): Whether to pad `y` such that frames are centered. Defaults to False.

        Returns:
            Tensor: The magnitude of the computed spectrogram.
        """

        y = y.squeeze(1)

        dtype_device = str(y.dtype) + "_" + str(y.device)
        wnsize_dtype_device = str(win_length) + "_" + dtype_device
        if wnsize_dtype_device not in self.hann_window:
            self.hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=self.hann_window[wnsize_dtype_device],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )

        spec = torch.view_as_real(spec)

        # Compute the magnitude
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

        return spec

    def wav_to_energy(
        self,
        y: torch.Tensor,
        n_fft: int,
        hop_length: int,
        win_length: int,
        center: bool = False
    ) -> torch.Tensor:
        r"""
        Convert a waveform to a spectrogram and compute the energy.

        Args:
            y (Tensor): The input waveform.
            n_fft (int): The FFT size.
            hop_length (int): The hop (stride) size.
            win_length (int): The window size.
            center (bool, optional): Whether to pad `y` such that frames are centered. Defaults to False.

        Returns:
            Tensor: The energy of the computed spectrogram.
        """

        spec = self.wav_to_spec(y, n_fft, hop_length, win_length, center=center)
        spec = torch.norm(spec, dim=1, keepdim=True).squeeze(0)

        # Normalize the energy
        return (spec - spec.mean()) / spec.std()

    def spec_to_mel(
            self, 
            spec: torch.Tensor, 
            n_fft: int, 
            num_mels: int, 
            sample_rate: int, 
            fmin: int, 
            fmax: int
    ) -> torch.Tensor:
        r"""
        Convert a spectrogram to a Mel spectrogram.

        Args:
            spec (torch.Tensor): The input spectrogram of shape [B, C, T].
            n_fft (int): The FFT size.
            num_mels (int): The number of Mel bands.
            sample_rate (int): The sample rate of the audio.
            fmin (int): The minimum frequency.
            fmax (int): The maximum frequency.

        Returns:
            torch.Tensor: The computed Mel spectrogram of shape [B, C, T].
        """

        mel_basis_key = self.name_mel_basis(spec, n_fft, fmax)

        if mel_basis_key not in self.mel_basis:
            mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
            self.mel_basis[mel_basis_key] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
        
        mel = torch.matmul(self.mel_basis[mel_basis_key], spec)
        mel = self.amp_to_db(mel)

        return mel

    def wav_to_mel(
        self,
        y: torch.Tensor,
        n_fft: int,
        num_mels: int, 
        sample_rate: int, 
        hop_length: int, 
        win_length: int, 
        fmin: int, 
        fmax: int, 
        center: bool = False
    ) -> torch.Tensor:
        r"""
        Convert a waveform to a Mel spectrogram.

        Args:
            y (torch.Tensor): The input waveform.
            n_fft (int): The FFT size.
            num_mels (int): The number of Mel bands.
            sample_rate (int): The sample rate of the audio.
            hop_length (int): The hop (stride) size.
            win_length (int): The window size.
            fmin (int): The minimum frequency.
            fmax (int): The maximum frequency.
            center (bool, optional): Whether to pad `y` such that frames are centered. Defaults to False.

        Returns:
            torch.Tensor: The computed Mel spectrogram.
        """

        # Convert the waveform to a spectrogram
        spec = self.wav_to_spec(y, n_fft, hop_length, win_length, center=center)

        # Convert the spectrogram to a Mel spectrogram
        mel = self.spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax)

        return mel
