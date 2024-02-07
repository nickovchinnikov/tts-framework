# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json

from IPython.core.display import HTML
from IPython.display import Audio, display
import pandas as pd
import torch

from training.modules import AcousticModule

# %%
checkpoint = "checkpoints/epoch=5209-step=536977.ckpt"
module = AcousticModule.load_from_checkpoint(checkpoint)
module.eval()


# %%
from training.modules import AcousticModule

module = AcousticModule()


# %%
dummy_text = "Once upon a time"
dummy_speaker_id = torch.tensor([0])

module.to_onnx(
    "./acoustic_module.onnx",
    (dummy_text, dummy_speaker_id),
    export_params=True,
    input_names = ["text", "speaker_id"],
    dynamic_axes={
        "text" : { 0 : "sequence_length" },
        "output" : { 0 : "batch_size" },
    },
    opset_version=17,
)


# %%
module.to_torchscript("./model.pt")


# %%
class WrapperModule(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, text: str, speaker_idx: torch.Tensor):
        return self.module(text, speaker_idx, lang="en")


# Wrap your model
wrapped_module = WrapperModule(module)

# Convert to TorchScript and save
script = torch.jit.script(wrapped_module)
torch.jit.save(script, "model.pt")

# %%
