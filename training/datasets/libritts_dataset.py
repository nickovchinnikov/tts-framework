from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchaudio import datasets

from model.config import lang2id
from training.preprocess import PreprocessLibriTTS
from training.tools import pad_1D, pad_2D, pad_3D


class LibriTTSDataset(Dataset):
    def __init__(
        self,
        root: str,
        batch_size: int,
        lang: str,
        sort: bool = False,
        drop_last: bool = False,
        download: bool = True,
    ):
        r"""A PyTorch dataset for loading preprocessed acoustic data.

        Args:
            root (str): Path to the directory where the dataset is found or downloaded.
            batch_size (int): Batch size for the dataset.
            lang (str): The language of the dataset.
            sort (bool, optional): Whether to sort the data by text length. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Defaults to False.
            download (bool, optional): Whether to download the dataset if it is not found. Defaults to True.
        """
        self.dataset = datasets.LIBRITTS(root=root, download=download)
        self.batch_size = batch_size

        self.preprocess_libtts = PreprocessLibriTTS(lang)

        self.sort = sort
        self.drop_last = drop_last

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

        data = self.preprocess_libtts(data)

        # TODO: bad way to do filtering, fix this!
        if data is None:
            print("Skipping due to preprocessing error")
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        data.wav = torch.FloatTensor(data.wav.unsqueeze(0))

        return {
            "id": data.utterance_id,
            "wav": data.wav,
            "mel": data.mel,
            "pitch": data.pitch,
            "text": data.phones,
            "attn_prior": data.attn_prior,
            "raw_text": data.raw_text,
            "normalized_text": data.normalized_text,
            "speaker": data.speaker_id,
            "pitch_is_normalized": data.pitch_is_normalized,
            # TODO: fix lang!
            "lang": lang2id["en"],
        }

    def collate_preprocess(self, data: List[Dict[str, Any]], idxs: List[int]) -> Tuple:
        r"""Reprocesses a batch of data samples.

        Args:
            data (List[Dict[str, Any]]): A list of data samples.
            idxs (List[int]): A list of indices for the samples to reprocess.

        Returns:
            Tuple: A tuple containing the reprocessed data.
        """
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

        return (
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            mels,
            pitches,
            pitches_stat,
            mel_lens,
            langs,
            attn_priors,
            wavs,
        )

    def collate_fn(self, data: List) -> List:
        r"""Collates a batch of data samples.

        Args:
            data (List): A list of data samples.

        Returns:
            List: A list of reprocessed data batches.
        """
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        return [self.collate_preprocess(data, idx) for idx in idx_arr]

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
