# %%
from torchaudio import datasets

dataset = datasets.LIBRITTS(
    root="../datasets_cache/LIBRITTS", download=True,
)

"""
Tensor:
    Waveform
int:
    Sample rate
str:
    Original text
str:
    Normalized text
int:
    Speaker ID
int:
    Chapter ID
str:
    Utterance ID
"""

# %%
# Tuple[Tensor, int, str, str, int, int, str]
dataset[0]

# %%
# ruff: noqa: E402
import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# %%
from dp.phonemizer import Phonemizer

checkpoint_ipa_forward = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "en_us_cmudict_ipa_forward.pt",
)
checkpoint_ipa_forward

# %%
phonemizer = Phonemizer.from_checkpoint(checkpoint_ipa_forward)
phonemes = phonemizer("Phonemizing an English text is imposimpable!", lang="en_us")
phonemes

# %%
from training.preprocess import PreprocessLibriTTS

preprocess = PreprocessLibriTTS()
preprocess

# %%
preprocess.acoustic(dataset[0])

# %%
from training.preprocess import NormalizeText

normilize_text = NormalizeText()

normilize_text(dataset[0][2])
# %%

# %%
# Read the file and extract the speaker IDs
with open("./libri_speakers.txt") as f:
    lines = f.readlines()[1:]  # Skip the header line
    speaker_ids = [int(line.split("|")[0].strip()) for line in lines]

# Create a mapping from original IDs to new IDs
id_mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(set(speaker_ids)))}

# Now you can use id_mapping to get a new ID for any old ID
# For example:
print(id_mapping[14])  # This will print 0

# %%
import json

# Save the id_mapping dictionary to a JSON file
with open("id_mapping.json", "w") as f:
    json.dump(id_mapping, f)

# %%
