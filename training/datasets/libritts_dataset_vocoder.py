from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset
from torchaudio import datasets

from training.preprocess import PreprocessLibriTTS
from training.tools import pad_1D, pad_2D


class LibriTTSDatasetVocoder(Dataset):
    r"""Loading preprocessed univnet model data."""

    def __init__(
        self,
        root: str,
        batch_size: int,
        download: bool = True,
    ):
        r"""A PyTorch dataset for loading preprocessed univnet data.

        Args:
            root (str): Path to the directory where the dataset is found or downloaded.
            batch_size (int): Batch size for the dataset.
            download (bool, optional): Whether to download the dataset if it is not found. Defaults to True.
        """
        self.dataset = datasets.LIBRITTS(root=root, download=download)
        self.batch_size = batch_size

        self.preprocess_libtts = PreprocessLibriTTS()

    def __len__(self) -> int:
        r"""Returns the number of samples in the dataset.

        Returns
            int: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r"""Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data.
        """
        # Retrive the dataset row
        data = self.dataset[idx]

        data = self.preprocess_libtts.univnet(data)

        if data is None:
            print("Skipping due to preprocessing error")
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        mel, audio, speaker_id = data

        return {
            "mel": mel,
            "audio": audio,
            "speaker_id": speaker_id,
        }

    def collate_fn(self, data: List) -> List:
        r"""Collates a batch of data samples.

        Args:
            data (List): A list of data samples.

        Returns:
            List: A list of reprocessed data batches.
        """
        data_size = len(data)

        idxs = list(range(data_size))

        # Initialize empty lists to store extracted values
        empty_lists: List[List] = [[] for _ in range(4)]
        (
            mels,
            mel_lens,
            audios,
            speaker_ids,
        ) = empty_lists

        # Extract fields from data dictionary and populate the lists
        for idx in idxs:
            data_entry = data[idx]

            mels.append(data_entry["mel"])
            mel_lens.append(data_entry["mel"].shape[1])
            audios.append(data_entry["audio"])
            speaker_ids.append(data_entry["speaker_id"])

        mels = torch.tensor(pad_2D(mels), dtype=torch.float32)
        mel_lens = torch.tensor(mel_lens, dtype=torch.int64)
        audios = torch.tensor(pad_1D(audios), dtype=torch.float32)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.int64)

        return [
            mels,
            mel_lens,
            audios,
            speaker_ids,
        ]
