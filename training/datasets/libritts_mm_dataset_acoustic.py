from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from training.tools import pad_1D, pad_2D, pad_3D

from .preprocessed_dataset import PreprocessedDataset


class LibriTTSMMDatasetAcoustic(Dataset):
    def __init__(self, file_path: str):
        r"""A PyTorch dataset for loading preprocessed acoustic data stored in memory-mapped files.

        Args:
            file_path (str): Path to the memory-mapped file.
        """
        self.data = torch.load(file_path)

    def __getitem__(self, idx: int):
        r"""Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data.
        """
        return self.data[idx]

    def __len__(self):
        r"""Returns the number of samples in the dataset.

        Returns
            int: Number of samples in the dataset.
        """
        return len(self.data)

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
        empty_lists: List[List] = [[] for _ in range(11)]
        (
            ids,
            speakers,
            texts,
            raw_texts,
            mels,
            pitches,
            attn_priors,
            langs,
            src_lens,
            mel_lens,
            wavs,
        ) = empty_lists

        # Extract fields from data dictionary and populate the lists
        for idx in idxs:
            data_entry = data[idx]
            ids.append(data_entry["id"])
            speakers.append(data_entry["speaker"])
            texts.append(data_entry["text"])
            raw_texts.append(data_entry["raw_text"])
            mels.append(data_entry["mel"])
            pitches.append(data_entry["pitch"])
            attn_priors.append(data_entry["attn_prior"].numpy())
            langs.append(data_entry["lang"])
            src_lens.append(data_entry["text"].shape[0])
            mel_lens.append(data_entry["mel"].shape[1])
            wavs.append(data_entry["wav"].numpy())

        # Convert langs, src_lens, and mel_lens to numpy arrays
        langs = np.array(langs)
        src_lens = np.array(src_lens)
        mel_lens = np.array(mel_lens)

        # NOTE: Instead of the pitches for the whole dataset, used stat for the batch
        # Take only min and max values for pitch
        pitches_stat = list(self.normalize_pitch(pitches)[:2])

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        attn_priors = pad_3D(attn_priors, len(idxs), max(src_lens), max(mel_lens))

        speakers = np.repeat(
            np.expand_dims(np.array(speakers), axis=1), texts.shape[1], axis=1,
        )
        langs = np.repeat(
            np.expand_dims(np.array(langs), axis=1), texts.shape[1], axis=1,
        )

        wavs = pad_2D(wavs)

        return [
            ids,
            raw_texts,
            torch.from_numpy(speakers),
            torch.from_numpy(texts).int(),
            torch.from_numpy(src_lens),
            torch.from_numpy(mels),
            torch.from_numpy(pitches),
            pitches_stat,
            torch.from_numpy(mel_lens),
            torch.from_numpy(langs),
            torch.from_numpy(attn_priors),
            torch.from_numpy(wavs),
        ]

    def normalize_pitch(
        self, pitches: List[torch.Tensor],
    ) -> Tuple[float, float, float, float]:
        r"""Normalizes the pitch values.

        Args:
            pitches (List[torch.Tensor]): A list of pitch values.

        Returns:
            Tuple: A tuple containing the normalized pitch values.
        """
        pitches_t = torch.concatenate(pitches)

        min_value = torch.min(pitches_t).item()
        max_value = torch.max(pitches_t).item()

        mean = torch.mean(pitches_t).item()
        std = torch.std(pitches_t).item()

        return min_value, max_value, mean, std

