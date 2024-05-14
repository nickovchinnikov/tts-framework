# %%
import json

import pandas as pd

from models.config.speakers import latest_selection

# %%
with open("training/datasets/speaker_id_mapping_libri.json") as f:
    id_mapping = json.load(f)

# Assuming `speaker_id_mapping` is your dictionary
id_mapping_back = {v: k for k, v in id_mapping.items()}
id_mapping_back

# %%
selected_speakers = set(map(id_mapping_back.get, latest_selection))
selected_speakers

# %%
speakers_df = pd.read_csv(
    "./datasets_cache/LIBRITTS/LibriTTS/speakers.tsv",
    sep="\t",
    names=["READER", "GENDER", "SUBSET", "NAME"],
)
speakers_df

# %%
speakers_df[speakers_df["READER"].isin(selected_speakers)]


# %%
latest_selection_ = set(map(str, selected_speakers))
latest_selection_

# %%
selected_speakers_filter = speakers_df["READER"].isin(latest_selection_)

selected_speakers_df = speakers_df[selected_speakers_filter]

(
    len(selected_speakers_df),
    len(latest_selection),
)

# %%
selected_speakers_dict = selected_speakers_df[["READER", "NAME"]].to_dict()
selected_speakers_dict

# %%
{
    value: int(key)
    for key, value in zip(
        map(id_mapping.get, selected_speakers_dict["READER"].values()),
        selected_speakers_dict["NAME"].values(),
    )
}


# %%
