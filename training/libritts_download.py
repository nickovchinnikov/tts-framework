# %%
import torchaudio.datasets as datasets

dataset = datasets.LIBRITTS(
    root="/mnt/Data/Projects/TTS experiments/LIBRITTS", download=True
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

preprocess = PreprocessLibriTTS(phonemizer, "english_only")
preprocess

# %%
preprocess(dataset[0])

# %%
