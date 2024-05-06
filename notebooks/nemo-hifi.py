# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from nemo.collections.tts.models import HifiGanModel

from models.vocoder.hifigan import HifiGan

# %%
hifigan_name = "tts_es_hifigan_ft_fastpitch_multispeaker"
# Load Vocoder
model = HifiGanModel.from_pretrained(hifigan_name)
model.generator.state_dict()

# %%
hifiGanModel = HifiGan()

hifiGanModel.generator.load_state_dict(model.generator.state_dict())
hifiGanModel.mpd.load_state_dict(model.discriminator.mpd.state_dict())
hifiGanModel.msd.load_state_dict(model.discriminator.msd.state_dict())

# %%
