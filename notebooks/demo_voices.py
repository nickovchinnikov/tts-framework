# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json

import torch
import pandas as pd

from IPython.core.display import HTML
from IPython.display import Audio, display

from training.modules import AcousticModule

# %%
checkpoint = "checkpoints/epoch=4988-step=484379.ckpt"
module = AcousticModule.load_from_checkpoint(checkpoint)

# %%
speakers_df = pd.read_csv("../datasets_cache/LIBRITTS/LibriTTS/speakers.tsv", sep="\t",)

speakers_libriid_speakerid = json.load(
    open("./speaker_id_mapping_libri.json", "r")
)

speakers_speakerid_libriid = {
    v: k for k, v in speakers_libriid_speakerid.items()
}
speakers_speakerid_libriid[0], speakers_libriid_speakerid["14"]

# %%
speakers_dict_train_clean_100 = speakers_df[speakers_df["SUBSET"] == "train-clean-100"].to_dict('records')
speakers_dict_train_clean_100[:10]


# %%
existed_speakers = []

for row in speakers_dict_train_clean_100:
    speaker_id = row["READER"]
    speaker_name = row["NAME"]

    existed_speakers.append(
        (
            speaker_id,
            speakers_libriid_speakerid[str(speaker_id)]
        )
    )

existed_speakers[:5]

# %%
from training.preprocess.tokenizer_ipa import TokenizerIPA

tokenizer = TokenizerIPA("en")

# %%
text = "Once upon a time, in the fantastical realm of Temeria, there lived a legendary figure known far and wide as the Gerald of Rivia. Gerald was no ordinary man! He was a Witcher, a monster hunter mutated and trained to combat supernatural threats that plagued the world."

text2 = "And he said: \"Hey!\", and I said: \"Hey, you!\""

text3 = "Once upon a time!"

_, tokens = tokenizer(text)
_, tokens2 = tokenizer(text2)

# %%
tokenizer.tokenizer.decode(tokens)

# %%
import time

# Initialize an empty string to store the HTML
html = "<table>"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for libriid, speaker_id in existed_speakers:
    start_time = time.time()
    with torch.no_grad():
        wav_prediction = module(
            text,
            torch.tensor([speaker_id]),
        )
    generation_time = time.time() - start_time

    speaker_data = speakers_df[speakers_df["READER"] == libriid]
    audio = Audio(wav_prediction, rate=22050)

    # Add a row to the HTML table
    html += "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2f} seconds</td></tr>".format(
        speaker_data.to_html(), audio._repr_html_(), device, generation_time
    )

    # Add a row to the HTML table
    # html += "<tr><td>{}</td><td>{}</td></tr>".format(speaker_data.to_html(), audio._repr_html_())

# Close the HTML table
html += "</table>"

# Display the HTML table
display(HTML(html))


# %%
speakers_speakerid_libriid[existed_speakers[2]]
# %%
speakers_df[speakers_df["READER"] == int(speakers_speakerid_libriid[existed_speakers[2]])]

# %%
for speaker in existed_speakers[:2]:
    with torch.no_grad():
        wav_prediction = module(
            text,
            torch.tensor([speaker]),
        )

    print(speakers_df[speakers_df["READER"] == speakers_speakerid_libriid[speaker]])

    Audio(wav_prediction, rate=22050)


# %%
import string
list(string.punctuation)

punct = ['!', '?', ',', '.', '-', ':', ';', '"', "'", '(', ')']

# %%
from dp.preprocessing.text import SequenceTokenizer

languages = ['de', 'en_us']
phoneme_symbols = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø', 'ŋ', 'œ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɹ', 'ɡ', 'ɪ', 'ʁ', 'ʃ', 'ʊ', 'ʌ', 'ʏ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̃', '̍', '̥', '̩', '̯', '͡', 'θ', '!', '?', ',', '.', '-', ':', ';', ' ', '"', "'", '(', ')']

phoneme_tokenizer = SequenceTokenizer(phoneme_symbols,
                                        languages=languages,
                                        lowercase=True,
                                        char_repeats=1,
                                        append_start_end=True)
tokens = phoneme_tokenizer("Hello, World!", 'en_us')
tokens

# %%
phoneme_tokenizer.decode(tokens)

# %%
'ˈ' == "'"
# %%
