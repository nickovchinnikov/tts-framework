from dataclasses import dataclass
import os
from typing import Any, List

import torch
from torch.utils.data import Dataset


@dataclass
class PreprocessedData:
    id: Any
    raw_text: Any
    speaker: Any
    text: Any
    src_len: Any
    mel: Any
    pitch: Any
    pitch_stat: Any
    mel_len: Any
    lang: Any
    attn_prior: Any
    wav: Any
    energy: Any


@dataclass
class PreprocessedDataset(Dataset):
    def __init__(self, cache_dir: str = "datasets_cache/LibriTTS_preprocessed"):
        self.cache_dir = cache_dir
        self.data = []

        for file in os.listdir(self.cache_dir):
            if file.endswith(".pt"):
                self.data.extend(torch.load(os.path.join(self.cache_dir, file)))

        for file in self.data_files:
            self.data.extend(torch.load(os.path.join(self.cache_dir, file)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
