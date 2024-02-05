# %%
# ruff: noqa: E402
import os

os.getcwd()

# LIBRITTS_PATH = "../../datasets_cache"
# os.environ["LIBRITTS_PATH"] = LIBRITTS_PATH
# print(LIBRITTS_PATH)
# from datasets import load_dataset
# print("Starting to load the dataset...")
# NOTE: can't download, error of the preprocessing...
# dataset = load_dataset('cdminix/libritts-r-aligned', data_dir="../../datasets_cache", split="train")
# %%
# dataset[0] # type: ignore
# %%
import pandas as pd

df = pd.read_csv("./libri_speakers.txt", sep="|")
df.head()

# %%
df.count() # 2484

# %%
print(df.columns)

# %%
df.columns = df.columns.str.strip()
df.columns

# %%
grouped = df.groupby("NAME")
grouped.head()

# %%
# 2477 unique speakers
grouped.count()

# %%
