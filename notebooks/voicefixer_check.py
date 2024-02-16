# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import IPython
import librosa
from voicefixer import Vocoder, VoiceFixer

voicefixer = VoiceFixer()
voicefixer


# %%
audio_array, _ = librosa.load("../results/example_360.wav", sr=22050)

result = voicefixer.restore_inmem(audio_array, mode=0)

IPython.display.Audio(result, rate=22050)

# %%

# %%

# %%
