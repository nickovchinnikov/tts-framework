import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import betabinom
import torch
from torch.utils.data import Dataset

from model.config import lang2id
from training.tools import pad_1D, pad_2D, pad_3D


class AcousticDataset(Dataset):
    def __init__(
        self,
        filename: str,
        batch_size: int,
        data_path: str,
        assets_path: str,
        is_eval: bool,
        sort: bool = False,
        drop_last: bool = False,
    ):
        r"""
        A PyTorch dataset for loading preprocessed acoustic data.

        Args:
            filename (str): Name of the metadata file.
            batch_size (int): Batch size for the dataset.
            data_path (str): Path to the preprocessed data.
            assets_path (str): Path to the assets directory.
            is_eval (bool): Whether the dataset is for evaluation or training.
            sort (bool, optional): Whether to sort the data by text length. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Defaults to False.
        """
        self.preprocessed_path = Path(data_path)
        self.batch_size = batch_size
        self.basename, self.speaker = self.process_meta(filename)
        with open(self.preprocessed_path / "speakers.json", encoding="utf-8") as f:
            self.speaker_map = json.load(f)
        self.sort = sort
        self.drop_last = drop_last
        self.is_eval = is_eval

    def __len__(self) -> int:
        r"""
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.basename)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r"""
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data.
        """
        basename = self.basename[idx]
        speaker_name = self.speaker[idx]
        speaker_id = self.speaker_map[speaker_name]
        data = torch.load(
            self.preprocessed_path / "data" / speaker_name / f"{basename}.pt"
        )
        raw_text = data["raw_text"]
        mel = data["mel"]
        pitch = data["pitch"]
        lang = data["lang"]
        phone = torch.LongTensor(data["phones"])
        attn_prior = self.beta_binomial_prior_distribution(
            phone.shape[0], mel.shape[1]
        ).T

        if mel.shape[1] < 20:
            print(
                "Skipping small sample due to the mel-spectrogram containing less than 20 frames"
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        sample = {
            "id": basename,
            "speaker_name": speaker_name,
            "speaker": speaker_id,
            "text": phone,
            "raw_text": raw_text,
            "mel": mel,
            "pitch": pitch,
            "lang": lang2id[lang],
            "attn_prior": attn_prior,
        }

        if phone.shape[0] >= mel.shape[1]:
            print(
                "Text is longer than mel, will be skipped due to monotonic alignment search ..."
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)

        if self.is_eval:
            data = torch.load(
                self.preprocessed_path / "wav" / speaker_name / f"{basename}.pt"
            )
            sample["wav"] = data["wav"].unsqueeze(0)

        return sample

    def process_meta(self, filename: str) -> Tuple[List[str], List[str]]:
        r"""
        Processes the metadata file and returns the basename and speaker lists.

        Args:
            filename (str): Name of the metadata file.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing the basename and speaker lists.
        """
        with open(self.preprocessed_path / filename, encoding="utf-8") as f:
            name = []
            speaker = []
            for line in f.readlines():
                n, s = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
        return name, speaker

    def beta_binomial_prior_distribution(
        self, phoneme_count: int, mel_count: int, scaling_factor: float = 1.0
    ) -> np.ndarray:
        r"""
        Computes the beta-binomial prior distribution for the attention mechanism.

        Args:
            phoneme_count (int): Number of phonemes in the input text.
            mel_count (int): Number of mel frames in the input mel-spectrogram.
            scaling_factor (float, optional): Scaling factor for the beta distribution. Defaults to 1.0.

        Returns:
            np.ndarray: A 2D numpy array containing the prior distribution.
        """
        P, M = phoneme_count, mel_count
        x = np.arange(0, P)
        mel_text_probs = []
        for i in range(1, M + 1):
            a, b = scaling_factor * i, scaling_factor * (M + 1 - i)
            rv = betabinom(P, a, b)
            mel_i_prob = rv.pmf(x)
            mel_text_probs.append(mel_i_prob)
        return np.array(mel_text_probs)

    def reprocess(self, data: List[Dict[str, Any]], idxs: List[int]) -> Tuple:
        r"""
        Reprocesses a batch of data samples.

        Args:
            data (List[Dict[str, Any]]): A list of data samples.
            idxs (List[int]): A list of indices for the samples to reprocess.

        Returns:
            Tuple: A tuple containing the reprocessed data.
        """
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        speaker_names = [data[idx]["speaker_name"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        pitches = [data[idx]["pitch"] for idx in idxs]
        langs = np.array([data[idx]["lang"] for idx in idxs])
        attn_priors = [data[idx]["attn_prior"] for idx in idxs]
        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[1] for mel in mels])

        texts = pad_1D(texts)
        mels = pad_2D(mels)
        pitches = pad_1D(pitches)
        attn_priors = pad_3D(attn_priors, len(idxs), max(text_lens), max(mel_lens))

        speakers = np.repeat(
            np.expand_dims(np.array(speakers), axis=1), texts.shape[1], axis=1
        )
        langs = np.repeat(
            np.expand_dims(np.array(langs), axis=1), texts.shape[1], axis=1
        )

        if self.is_eval:
            wavs = [data[idx]["wav"] for idx in idxs]
            wavs = pad_2D(wavs)
            return (
                ids,
                raw_texts,
                speakers,
                speaker_names,
                texts,
                text_lens,
                mels,
                pitches,
                mel_lens,
                langs,
                attn_priors,
                wavs,
            )
        else:
            return (
                ids,
                raw_texts,
                speakers,
                speaker_names,
                texts,
                text_lens,
                mels,
                pitches,
                mel_lens,
                langs,
                attn_priors,
            )

    def collate_fn(self, data: list) -> list:
        r"""
        Collates a batch of data samples.

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

        output = list()
        for idx in idx_arr:
            output.append(self.reprocess(data, idx))
        return output
