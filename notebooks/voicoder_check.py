# %%
import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import IPython

import torch

from dp.phonemizer import Phonemizer

from training.modules import AcousticDataModule, AcousticModule, VocoderModule
from training.preprocess.normalize_text import NormalizeText
from training.preprocess.tacotron_stft import TacotronSTFT

from training.preprocess.audio import safe_load


# %%
ckpt_acoustic="../checkpoints/epoch=516-step=100828.ckpt"
ckpt_vocoder="../checkpoints/vocoder.ckpt"
phonemizer_checkpoint = "../checkpoints/en_us_cmudict_ipa_forward.pt",

# %%
vocoder_module = VocoderModule.load_from_checkpoint(
    ckpt_vocoder,
)

module = AcousticModule.load_from_checkpoint(
    ckpt_acoustic,
    vocoder_module=vocoder_module,
)

# %%
text = "But the city itself began now to be visited too, I mean within the walls; but the number of people there were indeed extremely lessened by so great a multitude having been gone into the country; and even all this month of July they continued to flee, though not in such multitudes as formerly."

path_to_wav = "./26_496_000004_000000.wav"

params = { 
    "sampling_rate": 22050,
    "use_audio_normalization": True,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 100,
    "mel_fmin": 20,
    "mel_fmax": 11025,
}

tacotronSTFT = TacotronSTFT(
    filter_length=params["filter_length"],
    hop_length=params["hop_length"],
    win_length=params["win_length"],
    n_mel_channels=params["n_mel_channels"],
    sampling_rate=params["sampling_rate"],
    mel_fmin=params["mel_fmin"],
    mel_fmax=params["mel_fmax"],
    center=False,
)

wav, sr = safe_load(path_to_wav, params["sampling_rate"])
wav = torch.from_numpy(wav)

mel_spectrogram = tacotronSTFT.get_mel_from_wav(wav).unsqueeze(0)

result = vocoder_module.forward(mel_spectrogram)
result.shape

# %%
# add audio output
IPython.display.Audio(result.detach().cpu().numpy(), rate=sr*2) # type: ignore

# %%