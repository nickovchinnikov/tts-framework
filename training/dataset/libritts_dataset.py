from typing import Any, Dict, List, Tuple

from dp.phonemizer import Phonemizer
import numpy as np
import torch
from torch.utils.data import Dataset
import torchaudio.datasets as datasets

from model.config import PreprocessingConfig, PreprocessLangType, lang2id
from training.preprocess import PreprocessLibriTTS
from training.tools import pad_1D, pad_2D, pad_3D


class LibriTTSDataset(Dataset):
    def __init__(
        self,
        root: str,
        batch_size: int,
        is_eval: bool,
        phonemizer: Phonemizer,
        processing_lang_type: PreprocessLangType = "english_only",
        sort: bool = False,
        drop_last: bool = False,
        download: bool = True,
    ):
        r"""
        A PyTorch dataset for loading preprocessed acoustic data.

        Args:
            root (str): Path to the directory where the dataset is found or downloaded.
            batch_size (int): Batch size for the dataset.
            is_eval (bool): Whether the dataset is for evaluation or training.
            phonemizer (Phonemizer): The g2p phonemizer.
            processing_lang_type (PreprocessLangType, optional): The preprocessing language type. Defaults to "english_only".
            sort (bool, optional): Whether to sort the data by text length. Defaults to False.
            drop_last (bool, optional): Whether to drop the last batch if it is smaller than the batch size. Defaults to False.
            download (bool, optional): Whether to download the dataset if it is not found. Defaults to True.
        """
        self.dataset = datasets.LIBRITTS(root=root, download=download)
        self.batch_size = batch_size

        self.phonemizer = phonemizer

        self.preprocess_config = PreprocessingConfig(processing_lang_type)
        self.preprocess_libtts = PreprocessLibriTTS(self.phonemizer, processing_lang_type)

        self.sort = sort
        self.drop_last = drop_last
        self.is_eval = is_eval

    def __len__(self) -> int:
        r"""
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        r"""
        Returns a sample from the dataset at the given index.

        Args:
            idx (int): Index of the sample to return.

        Returns:
            Dict[str, Any]: A dictionary containing the sample data.
        """

        # Retrive the dataset row
        data = self.dataset[idx]

        data = self.preprocess_libtts(data)

        if data.mel.shape[1] < 20:
            print(
                "Skipping small sample due to the mel-spectrogram containing less than 20 frames"
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)
        
        if data.phone.shape[0] >= data.mel.shape[1]:
            print(
                "Text is longer than mel, will be skipped due to monotonic alignment search ..."
            )
            rand_idx = np.random.randint(0, self.__len__())
            return self.__getitem__(rand_idx)
        
        if self.is_eval:
            data.wav = data.wav.unsqueeze(0)

        sample = {
            "id": data.utterance_id,
            "speaker": data.speaker_id,
            "text": data.phones,
            "raw_text": data.raw_text,
            "mel": data.mel,
            "pitch": data.pitch,
            # TODO: fix lang!
            "lang": lang2id["en"],
            "attn_prior": data.attn_prior,
        }

        return sample

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
