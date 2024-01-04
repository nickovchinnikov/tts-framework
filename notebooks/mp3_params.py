# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch

from pydub import AudioSegment
from pydub.utils import mediainfo

from IPython.display import Audio

from training.modules.acoustic_module import AcousticModule
from server.utils import speakers_info, returnAudioBuffer, sentences_split, audio_package

# %%
info = mediainfo("./response_valid.mp3")
info

# %%
checkpoint = "../checkpoints/epoch=5482-step=601951.ckpt"

device_gpu = torch.device("cuda")
module_gpu = AcousticModule.load_from_checkpoint(checkpoint).to(device_gpu)

# %%
SAMPLING_RATE = 22050

text = "Hello, my name is John. I am a student at the University of California, Berkeley. I am majoring in Computer Science. I am a member of the Berkeley Artif."

speaker = torch.tensor([122], device=device_gpu)

with torch.no_grad():
    wav_prediction = module_gpu(
        text,
        speaker,
    ).detach().cpu().numpy()

    audio = AudioSegment(
        wav_prediction.tobytes(),
        frame_rate=SAMPLING_RATE,
        sample_width=wav_prediction.dtype.itemsize,
        channels=1
    )

# %%
audio.export("./response.mp3", format="mp3")

# %%
from pydub.playback import play
play(audio)

# %%
