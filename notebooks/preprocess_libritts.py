# %%
import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json

import torch
from torch.utils.data import Dataset
from torchaudio import datasets
from tqdm import tqdm

from models.config import lang2id
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

preprocess_libtts = PreprocessLibriTTS(
    lang,
    phonemizer_checkpoint="checkpoints/en_us_cmudict_ipa_forward.pt",
)
dataset = datasets.LIBRITTS(root=root, download=True)

# Load the id_mapping dictionary from the JSON file
with open("speaker_id_mapping_libri.json") as f:
    id_mapping = json.load(f)

# %%
# Preprocess all samples and store them in lists
data = []
pbar = tqdm(total=len(dataset))

for sample in dataset:
    preprocessed_sample = preprocess_libtts.acoustic(sample)

    if preprocessed_sample is not None:
        preprocessed_sample.wav = torch.FloatTensor(preprocessed_sample.wav.unsqueeze(0))
        res = {
            "id": preprocessed_sample.utterance_id,
            "wav": preprocessed_sample.wav,
            "mel": preprocessed_sample.mel,
            "pitch": preprocessed_sample.pitch,
            "text": preprocessed_sample.phones,
            "attn_prior": preprocessed_sample.attn_prior,
            "raw_text": preprocessed_sample.raw_text,
            "normalized_text": preprocessed_sample.normalized_text,
            "speaker": id_mapping.get(str(preprocessed_sample.speaker_id)),
            "pitch_is_normalized": preprocessed_sample.pitch_is_normalized,
            # TODO: fix lang!
            "lang": lang2id["en"],
        }
        data.append(res)
    pbar.update(1)

# Create a PreprocessedDataset from the list
preprocessed_dataset = PreprocessedDataset(data)

len(preprocessed_dataset)

# %%
# Save the PreprocessedDataset to disk
torch.save(
    preprocessed_dataset,
    "checkpoints/libri_preprocessed_data.pt",
)

# %%
from torch.utils.data import DataLoader

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
