# %%
# %%
import os
from pathlib import Path
import sys

from lightning.pytorch.core import LightningModule

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(Path(SCRIPT_DIR).parent))

from IPython import display
import torchaudio
from voicefixer import Vocoder

from models.vocoder.univnet import UnivNet
from training.datasets.hifi_libri_dataset import HifiLibriDataset, HifiLibriItem

# %%
vocoder_vf = Vocoder(44100)
dataset = HifiLibriDataset(
    root="../datasets_cache",
    cache_dir="../datasets_cache",
    cache=True,
)

# %%
vocoder_un = UnivNet()
vocoder_un

# %%
item = dataset[0]
wav = vocoder_vf.forward(item.mel.permute((1, 0)).unsqueeze(0))

display.Audio(wav.squeeze(0).cpu().detach().numpy(), rate=44100)

# %%
