import math
from pathlib import Path
import random
from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models.config import HifiGanPretrainingConfig

from .hifi_libri_dataset import NUM_JOBS, HifiLibriDataset


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
            max_seconds=25.0,  # To be sure that all the audio files from the dataset will be used
            include_libri=False,  # Exclude LibriTTS dataset
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

    def __getitem__(self, idx: int) -> Tuple[str, Tensor, Tensor]:
        r"""Get an item from the dataset.

        If caching is enabled and the item is in the cache, the cached item is returned.
        Otherwise, the item is loaded from the dataset, preprocessed, and returned.

        Args:
            idx (int): The index of the item in the dataset.

        Returns:
            Tuple[str, Tensor, Tensor]: The ID of the item, the audio waveform, and the mel spectrogram.
        """
        cache_file = self.get_cache_file_path(idx)

        if self.cache and cache_file.exists():
            cached_data: Tuple[str, Tensor, Tensor] = torch.load(cache_file)
            return cached_data

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

        result = (item.id, audio, mel.squeeze(0))

        if self.cache:
            # Create the cache subdirectory if it doesn't exist
            Path.mkdir(
                self.get_cache_subdir_path(idx),
                parents=True,
                exist_ok=True,
            )
            # Save the preprocessed data to the cache
            torch.save(result, cache_file)

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


def train_dataloader(
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
    num_workers: int = 0,
    shuffle: bool = False,
    batch_size: int = 5,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    r"""Create a DataLoader for the training data.

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
        num_workers (int, optional): The number of worker processes to use for loading the data. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        batch_size (int, optional): The batch size. Defaults to 5.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch. Defaults to True.
        num_gpus (int, optional): The number of GPUs to use. Defaults to 1.

    Returns:
        DataLoader: A DataLoader for the training data.
    """
    trainset = HifiGanDataset(
        lang=lang,
        root=root,
        sampling_rate=sampling_rate,
        hifitts_path=hifitts_path,
        hifi_cutset_file_name=hifi_cutset_file_name,
        libritts_path=libritts_path,
        libritts_cutset_file_name=libritts_cutset_file_name,
        libritts_subsets=libritts_subsets,
        cache=cache,
        cache_dir=cache_dir,
        num_jobs=num_jobs,
    )

    train_loader = DataLoader(
        trainset,
        num_workers=num_workers,
        shuffle=shuffle,
        batch_size=batch_size,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

    return train_loader
