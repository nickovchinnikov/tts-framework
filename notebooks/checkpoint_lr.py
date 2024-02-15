# %%
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# %%

# Load the state dictionary from the checkpoint
checkpoint = torch.load("./checkpoints/logs_new_training_libri-360_epoch=1139-step=319899.ckpt")
checkpoint.keys()

# %%
from models.tts.delightful_tts.delightful_tts import DelightfulTTS

# Load the model
model = DelightfulTTS()

# Get the model's state dictionary
model_state_dict = model.state_dict()

model_state_dict.keys()

# %%
from models.tts.delightful_tts.delightful_tts import DelightfulTTS

DelightfulTTS.load_from_checkpoint("./checkpoints/logs_new_training_libri-360_epoch=1165-step=326217.ckpt", strict=False)

# %%

model_state_dict["acoustic_model.energy_adaptor.energy_predictor.layers.0.conv.pointwise.conv.weight"].shape

# %%
checkpoint_state_dict = checkpoint["state_dict"]

missing_keys = ["acoustic_model.energy_adaptor.energy_predictor.layers.0.conv.pointwise.conv.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.0.conv.pointwise.conv.bias", "acoustic_model.energy_adaptor.energy_predictor.layers.0.conv.depthwise.conv.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.0.conv.depthwise.conv.bias", "acoustic_model.energy_adaptor.energy_predictor.layers.2.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.2.bias", "acoustic_model.energy_adaptor.energy_predictor.layers.4.conv.pointwise.conv.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.4.conv.pointwise.conv.bias", "acoustic_model.energy_adaptor.energy_predictor.layers.4.conv.depthwise.conv.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.4.conv.depthwise.conv.bias", "acoustic_model.energy_adaptor.energy_predictor.layers.6.weight", "acoustic_model.energy_adaptor.energy_predictor.layers.6.bias", "acoustic_model.energy_adaptor.energy_predictor.linear_layer.weight", "acoustic_model.energy_adaptor.energy_predictor.linear_layer.bias", "acoustic_model.energy_adaptor.energy_emb.weight", "acoustic_model.energy_adaptor.energy_emb.bias",
]

for key in missing_keys:
    checkpoint_state_dict[key] = model_state_dict[key]

checkpoint_state_dict["acoustic_model.energy_adaptor.energy_emb.bias"].shape


# %%
checkpoint["state_dict"] = checkpoint_state_dict
torch.save(checkpoint, "./checkpoints/new_training_libri-360_epoch=1139-step=319899_fixed.ckpt")



# %%
import torch

path ="./epoch=4677-step=410361.ckpt"

# Load checkpoint
ckpt = torch.load(path, map_location=torch.device("cpu"))
ckpt

# %%
ckpt["optimizer_states"][0]["param_groups"][0]["lr"]

# %%
ckpt["optimizer_states"][0]["param_groups"][0]["initial_lr"]

# %%
path2 = "epoch=121-step=7808.ckpt"
# Load checkpoint 2
ckpt2 = torch.load(path2, map_location=torch.device("cpu"))

# %%
latest_lr = ckpt2["optimizer_states"][0]["param_groups"][0]["lr"]
latest_lr

# %%
ckpt2["optimizer_states"][0]["param_groups"][0]["lr"] = latest_lr * 2000

ckpt2["optimizer_states"][0]["param_groups"][0]["lr"]

# %%
ckpt2["optimizer_states"][0]["param_groups"][0]["initial_lr"]

# %%
path3 = "./epoch=121-step=7808_fixedlr.ckpt"

# Save the new checkpoint
torch.save(ckpt2, path3)

# %%
