from dataclasses import asdict, dataclass
import math
from pathlib import Path
import random
from typing import Dict, List

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset

from models.config import HifiGanPretrainingConfig

from .hifi_libri_dataset import DATASET_TYPES, NUM_JOBS, HifiLibriDataset


@dataclass
class HifiGANItem:
    r"""Dataset row for the HiFiTTS and LibriTTS datasets combined in this code.
    Prepared for the HiFi-GAN model training.

    Args:
        id (str): The ID of the item.
        wav (Tensor): The waveform of the audio.
        mel (Tensor): The mel spectrogram.
        speaker (int): The speaker ID.
        lang (int): The language ID.
        dataset_type (DATASET_TYPES): The type of dataset.
    """

    id: str
    wav: Tensor
    mel: Tensor
    speaker: int
    lang: int
    dataset_type: DATASET_TYPES


class HifiGanDataset(Dataset):
    r"""A PyTorch Dataset for the HiFi-GAN model.

    Args:
        lang (str, optional): The language of the dataset. Defaults to "en".
        root (str, optional): The root directory of the dataset. Defaults to "datasets_cache".
        sampling_rate (int, optional): The sampling rate of the audio. Defaults to 44100.
        hifitts_path (str, optional): The path to the HiFiTTS dataset. Defaults to "hifitts".
        hifi_cutset_file_name (str, optional): The file name of the HiFiTTS cutset. Defaults to "hifi.json.gz".
        libritts_path (str, optional): The path to the LibriTTS dataset. Defaults to "librittsr".
        libritts_cutset_file_name (str, optional): The file name of the LibriTTS cutset. Defaults to "libri.json.gz".
        libritts_subsets (Union[List[str], str], optional): The subsets of the LibriTTS dataset to use. Defaults to "all".
        cache (bool, optional): Whether to cache the dataset. Defaults to False.
        cache_dir (str, optional): The directory to cache the dataset in. Defaults to "/dev/shm".
        num_jobs (int, optional): The number of jobs to use for preparing the dataset. Defaults to NUM_JOBS.
    """

    def __init__(
        self,
        lang: str = "en",
        root: str = "datasets_cache",
        sampling_rate: int = 44100,
        hifitts_path: str = "hifitts",
        hifi_cutset_file_name: str = "hifi.json.gz",
        libritts_path: str = "librittsr",
        libritts_cutset_file_name: str = "libri.json.gz",
        libritts_subsets: List[str] | str = "all",
        cache: bool = False,
        cache_dir: str = "/dev/shm",
        num_jobs: int = NUM_JOBS,
    ):
        self.cache = cache
        self.cache_dir = Path(cache_dir) / "cache-hifigan-dataset"

        self.pretraining_config = HifiGanPretrainingConfig()

        self.dataset = HifiLibriDataset(
            lang=lang,
            root=root,
            sampling_rate=sampling_rate,
            hifitts_path=hifitts_path,
            hifi_cutset_file_name=hifi_cutset_file_name,
            libritts_path=libritts_path,
            libritts_cutset_file_name=libritts_cutset_file_name,
            libritts_subsets=libritts_subsets,
            num_jobs=num_jobs,
        )

        self.segment_size = self.pretraining_config.segment_size
        self.hop_size = self.dataset.preprocess_config.stft.hop_length

    def __len__(self):
        r"""Get the number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.dataset)

    def get_cache_subdir_path(self, idx: int) -> Path:
        r"""Calculate the path to the cache subdirectory.

        Args:
            idx (int): The index of the cache subdirectory.

        Returns:
            Path: The path to the cache subdirectory.
        """
        return self.cache_dir / str(((idx // 1000) + 1) * 1000)

    def get_cache_file_path(self, idx: int) -> Path:
        r"""Calculate the path to the cache file.

        Args:
            idx (int): The index of the cache file.

        Returns:
            Path: The path to the cache file.
        """
        return self.get_cache_subdir_path(idx) / f"{idx}.pt"

    def __getitem__(self, idx: int) -> HifiGANItem:
        r"""Get an item from the dataset.

        If caching is enabled and the item is in the cache, the cached item is returned.
        Otherwise, the item is loaded from the dataset, preprocessed, and returned.

        Args:
            idx (int): The index of the item in the dataset.

        Returns:
            HifiGANItem: The preprocessed item from the dataset.
        """
        cache_file = self.get_cache_file_path(idx)

        if self.cache and cache_file.exists():
            cached_data: Dict = torch.load(cache_file)
            # Cast the cached data to the PreprocessForAcousticResult class
            result = HifiGANItem(**cached_data)
            return result

        item = self.dataset[idx]
        frames_per_seg = math.ceil(self.segment_size / self.hop_size)

        audio = item.wav
        mel = item.mel.unsqueeze(0)

        if audio.size(1) >= self.segment_size:
            mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)  # noqa: S311
            mel = mel[:, :, mel_start : mel_start + frames_per_seg]
            audio = audio[
                :,
                mel_start * self.hop_size : (mel_start + frames_per_seg)
                * self.hop_size,
            ]
        else:
            mel = F.pad(
                mel,
                (0, frames_per_seg - mel.size(2)),
                "constant",
            )
            audio = F.pad(
                audio,
                (0, self.segment_size - audio.size(1)),
                "constant",
            )

        result = HifiGANItem(
            id=item.id,
            wav=audio,
            mel=mel,
            speaker=item.speaker,
            lang=item.lang,
            dataset_type=item.dataset_type,
        )

        if self.cache:
            # Create the cache subdirectory if it doesn't exist
            Path.mkdir(
                self.get_cache_subdir_path(idx),
                parents=True,
                exist_ok=True,
            )
            # Save the preprocessed data to the cache
            torch.save(asdict(result), cache_file)

        return result

    def __iter__(self):
        r"""Method makes the class iterable. It iterates over the `_walker` attribute
        and for each item, it gets the corresponding item from the dataset using the
        `__getitem__` method.

        Yields:
            The item from the dataset corresponding to the current item in `_walker`.
        """
        for item in range(self.__len__()):
            yield self.__getitem__(item)
