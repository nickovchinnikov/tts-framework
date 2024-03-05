# %%
import os
import sys

import torch
import torchaudio

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from models.generators.delightful_univnet import DelightfulUnivnet
from models.tts.delightful_tts.delightful_tts_refined import DelightfulTTS
from models.vocoder.univnet import UnivNet

# %%

delightful_tts = DelightfulTTS.load_from_checkpoint(
    checkpoint_path="./checkpoints/logs_new_training_libri-360-swa_epoch=274-step=35326.ckpt",
    strict=False,
)

univnet = UnivNet.load_from_checkpoint(
    checkpoint_path="./checkpoints/logs_new_training_vocoder_libri_epoch=19-step=9680.ckpt",
    strict=False,
)

# %%
delightful_univnet = DelightfulUnivnet()

# Load state dict
delightful_univnet.acoustic_model.load_state_dict(delightful_tts.acoustic_model.state_dict())
delightful_univnet.univnet.load_state_dict(univnet.univnet.state_dict())
delightful_univnet.discriminator.load_state_dict(univnet.discriminator.state_dict())

# # %%
# Save the state of DelightfulUnivnet
# torch.save(delightful_univnet.state_dict(), "delightful_univnet_state.pth")

# # %%
# # Load the saved state of DelightfulUnivnet
# delightful_univnet = DelightfulUnivnet()
# delightful_univnet.load_state_dict(torch.load("delightful_univnet_state.pth"))
# # delightful_univnet

# %%
text = "Hello, this is a test sentence."
speaker = torch.tensor([5])

wav_prediction = delightful_univnet.forward(
    text,
    speaker,
)

# Save the audio to a file
torchaudio.save(        # type: ignore
    "../results/output2_.wav",
    wav_prediction.squeeze(0).detach().cpu(),
    22050,
)

# %%
