# %%
import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json

import torch
from torch.utils.data import DataLoader, Dataset
from torchaudio import datasets
from tqdm import tqdm

from models.config import lang2id
from training.datasets.libritts_dataset_acoustic import LibriTTSDatasetAcoustic
from training.datasets.libritts_r import LIBRITTS_R
from training.preprocess import PreprocessLibriTTS


# %%
class PreprocessedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# %%
root: str = "datasets_cache/LIBRITTS"
lang: str = "en"
url: str = "train-clean-360"

dataset = LibriTTSDatasetAcoustic(root=root, url=url, download=True)

# %%
batch_size = 8
data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    persistent_workers=True,
    pin_memory=True,
    num_workers=batch_size,
    collate_fn=dataset.collate_fn,
)

data = []
for i, batch in enumerate(tqdm(data_loader)):
    data.extend(batch)
    if (i + 1) % (10000 // batch_size) == 0:
        torch.save(
            data,
            f"datasets_cache/{url}_step_{i+1}_preprocessed.pt",
        )
        data = []

# Save the remaining data
if data:
    torch.save(
        data,
        f"datasets_cache/{url}_step_99_preprocessed.pt",
    )

# %%
from multiprocessing import Pool

# Preprocess all samples and store them in lists
data = []


def process_sample(preprocessed_sample):
    return preprocessed_sample
    # preprocessed_sample = preprocess_libtts.acoustic(sample)


with Pool(processes=10) as p:
    # data += list(tqdm(p.imap(process_sample, dataset), total=len(dataset)))
    for i, result in enumerate(
        tqdm(p.imap(process_sample, dataset), total=len(dataset)),
    ):
        data.append(result)
        if (i + 1) % 20000 == 0:
            torch.save(
                data,
                f"datasets_cache/{url}_preprocessed.pt",
            )
            data = []


# %%
# Save the PreprocessedDataset to disk
torch.save(
    data,
    f"datasets_cache/{url}_preprocessed.pt",
)

# %%
# Preprocess all samples and store them in lists
data = []
pbar = tqdm(total=len(dataset))

for sample in dataset:
    preprocessed_sample = preprocess_libtts.acoustic(sample)

    if preprocessed_sample is not None:
        preprocessed_sample.wav = torch.FloatTensor(
            preprocessed_sample.wav.unsqueeze(0),
        )
        res = {
            "id": preprocessed_sample.utterance_id,
            "wav": preprocessed_sample.wav,
            "mel": preprocessed_sample.mel,
            "pitch": preprocessed_sample.pitch,
            "text": preprocessed_sample.phones,
            "attn_prior": preprocessed_sample.attn_prior,
            "energy": preprocessed_sample.energy,
            "raw_text": preprocessed_sample.raw_text,
            "normalized_text": preprocessed_sample.normalized_text,
            "speaker": id_mapping.get(str(preprocessed_sample.speaker_id)),
            "pitch_is_normalized": preprocessed_sample.pitch_is_normalized,
            # TODO: fix lang!
            "lang": lang2id["en"],
        }
        data.append(res)
    pbar.update(1)

# %%
# Save the PreprocessedDataset to disk
torch.save(
    data,
    f"{url}_preprocessed.pt",
)
# %%

# Create a PreprocessedDataset from the list
preprocessed_dataset = PreprocessedDataset(data)

len(preprocessed_dataset)

# %%
torch.save([1, 2, 3], "test.pt")

# %%
torch.load("test.pt")

# %%
# Save the PreprocessedDataset to disk
torch.save(
    preprocessed_dataset,
    "checkpoints/libri_preprocessed_data.pt",
)

# %%

preprocessed_dataset = torch.load(
    "checkpoints/libri_preprocessed_data.pt",
)

# Create a DataLoader
batch_size = 32  # Set the batch size to the desired value
data_loader = DataLoader(
    preprocessed_dataset,
    batch_size=batch_size,
    shuffle=True,
)

for batch in data_loader:
    print(batch)
    break
