# %%
import os
import sys

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# %%
# ruff: noqa: E402
from acoustic_model import AcousticModel
import torch
from univnet import Generator as UnivNet

from model.config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
    VocoderModelConfig,
)
from model.helpers import get_device

# %%
checkpoint_base = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "assets",
    "v0.1.0",
)
checkpoint_base

# %%
# Acoustic checkpoint loading
checkpoint_acoustic_path = os.path.join(
    checkpoint_base,
    "acoustic_pretrained.pt",
)
ckpt_acoustic = torch.load(checkpoint_acoustic_path)
ckpt_acoustic

# %%


# %%

model_config = AcousticENModelConfig()
preprocess_config = PreprocessingConfig("english_only")
acoustic_pretraining_config = AcousticPretrainingConfig()

data_path = os.path.join(
    SCRIPT_DIR,
    "config",
)

device = get_device()

model = AcousticModel(
    data_path,
    preprocess_config,
    model_config,
    fine_tuning=True,
    # Import from the checkpoint
    n_speakers=5392,
    device=device,
)
model

# %%
# Result is not so bad, it works, but the model is not the same
# I found where the problem is, it's in the Conformer class
# Error when loading the model weights
# self.decoder = Conformer(
#     dim=model_config.decoder.n_hidden,
#     n_layers=model_config.decoder.n_layers,
#     n_heads=model_config.decoder.n_heads,
#     embedding_dim=model_config.speaker_embed_dim,
#     # There is shouldn't be the lang_embed_dim, and it works...
#     # Need to check the inference
#     # embedding_dim=model_config.speaker_embed_dim + model_config.lang_embed_dim,
#     p_dropout=model_config.decoder.p_dropout,
#     kernel_size_conv_mod=model_config.decoder.kernel_size_conv_mod,
#     with_ff=model_config.decoder.with_ff,
#     device=self.device,
# )
# TODO: check this out: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/13
# https://stackoverflow.com/a/65065854/10828885
# Or you can del the broken weight, example: ckpt["gen"]["speaker_embed"]
model.load_state_dict(ckpt_acoustic["gen"], strict=False)
model

# %%
# Voicoder checkpoint loading
checkpoint_voicoder_path = os.path.join(
    checkpoint_base,
    "vocoder_pretrained.pt",
)
checkpoint_voicoder = torch.load(checkpoint_voicoder_path)
checkpoint_voicoder

# %%
voicoder_model_conf = VocoderModelConfig()

univnet = UnivNet(voicoder_model_conf, preprocess_config, device=device)
univnet

# %%
# Done, it works great!
univnet.load_state_dict(checkpoint_voicoder["generator"], strict=False)
univnet

# %%
symbols = [str(el) for el in range(256)]
symbols
# %%
symbol2id = {s: i for i, s in enumerate(symbols)}
symbol2id
# %%
{i: s for i, s in enumerate(symbols)}

# %%
symbol2id["."]
