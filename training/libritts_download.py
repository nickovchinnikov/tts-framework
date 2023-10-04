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
