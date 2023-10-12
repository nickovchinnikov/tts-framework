# %%
# ruff: noqa: E402
import os

LIBRITTS_PATH = "../../datasets_cache"

os.environ["LIBRITTS_PATH"] = LIBRITTS_PATH

print(LIBRITTS_PATH)

from datasets import load_dataset

dataset = load_dataset('cdminix/libritts-r-aligned', data_dir="../../datasets_cache", split="train")

# %%
dataset[0]

# %%

# %%

# %%
