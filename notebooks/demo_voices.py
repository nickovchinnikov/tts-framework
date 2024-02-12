# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import json
import time

from IPython.core.display import HTML
from IPython.display import Audio, display
import pandas as pd
import torch

# from models.univnet import Discriminator, UnivNet
from training.modules import AcousticModule, VocoderModule

# %%
checkpoint = "../checkpoints/epoch=570-step=180582.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

module = AcousticModule.load_from_checkpoint(checkpoint).to(device)

module.eval()

# %%
univnet = VocoderModule.load_from_checkpoint("./checkpoints/vocoder.ckpt")

univnet.eval()

# %%
speakers_df = pd.read_csv("../datasets_cache/LIBRITTS/LibriTTS/speakers.tsv", sep="\t")

speakers_libriid_speakerid = json.load(
    open("./speaker_id_mapping_libri.json"),
)

speakers_speakerid_libriid = {
    v: k for k, v in speakers_libriid_speakerid.items()
}
speakers_speakerid_libriid[0], speakers_libriid_speakerid["14"]


# %%
speakers_dict_train_clean_100 = speakers_df[speakers_df["SUBSET"] == "train-clean-100"].to_dict("records")
speakers_dict_train_clean_100[:10]


# %%
ids = {32, 102, 122, 160, 167, 487, 2419, 1641, 1315, 119, 155, 156, 197, 206, 208, 359, 360, 396, 426, 88, 99, 216, 192}
existed_speakers = []

for row in speakers_dict_train_clean_100:
    speaker_id = row["READER"]

    internal_id = speakers_libriid_speakerid[str(speaker_id)]

    speaker_name = row["NAME"]
    gender = row["GENDER"]

    # if internal_id not in ids:
    #     continue

    existed_speakers.append(
        (
            speaker_id,
            internal_id,
            speaker_name,
            gender,
        ),
    )

existed_speakers[:5]


# %%
def gen_table(text, existed_speakers):
    # Initialize an empty string to store the HTML
    html = "<table>"

    html += f"<h4>{text}</h4>"

    html += "<tr><th>Speaker</th><th>Audio</th><th>Device</th><th>Generation Time</th></tr>"

    for libriid, speaker_id, speaker_name, gender in existed_speakers:
        start_time = time.time()
        with torch.no_grad():
            wav_prediction = module(
                text,
                torch.tensor([speaker_id], device=device),
            )
        generation_time = time.time() - start_time

        speaker_data = speakers_df[speakers_df["READER"] == libriid]
        audio = Audio(wav_prediction.cpu(), rate=22050)

        # Add a row to the HTML table
        html += "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:.2f} seconds</td></tr>".format(
            speaker_data.to_html(), audio._repr_html_(), device, generation_time,
        )

        # Add a row to the HTML table
        # html += "<tr><td>{}</td><td>{}</td></tr>".format(speaker_data.to_html(), audio._repr_html_())

    # Close the HTML table
    html += "</table>"

    return HTML(html)

# %%
text = "As the snake shook its head, a deafening shout behind Harry made both of them jump.'DUDLEY! MR DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON'T BELIEVE WHAT IT'S DOING!'Dudley came waddling towards them as fast as he could. 'Out of the way, you,' he said, punching Harry in the ribs. Caught by surprise, Harry fell hard on the concrete floor. What came next happened so fast no one saw how it happened – one second, Piers and Dudley were leaning right up close to the glass, the next, they had leapt back with howls of horror."

# Display the HTML table
display(gen_table(text, existed_speakers))

# %%
text = "Governments influence a surprising amount of literature. Some of it pretty good “ALL ART is propaganda”, wrote George Orwell in 1940, “but not all propaganda is art.” Few people would argue with the second part of that aphorism. There is nothing artistic about the dreadful ramblings of “Mein Kampf”. But the first seems true only if you are using a broad definition of propaganda. These days great works of art rarely set out to serve the purposes of a government. They may promote causes, but that is not normally why people esteem them. The books on this list, however, partially vindicate the first part of Orwell’s assertion. Governments or ideological groups either encouraged their authors to write them or promoted their writings for political ends. During the cold war Western intelligence agencies subsidised authors, sometimes very good ones. The CIA set up literary magazines in France, Japan and Africa. One purpose was to counter censorship by autocrats. Another was to make global culture friendlier to Western aims. British intelligence services commissioned works of fiction that supported empire. Some writers consciously offered their pens to the state; others did not realise that governments or groups would promote their work. Here are six books, all by authors of merit, that are works of propaganda in one way or another."

# Display the HTML table
display(gen_table(text, existed_speakers))

# %%
from training.preprocess.tokenizer_ipa import TokenizerIPA

tokenizer = TokenizerIPA("en")

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

punct = ["!", "?", ",", ".", "-", ":", ";", '"', "'", "(", ")"]

# %%
from dp.preprocessing.text import SequenceTokenizer

languages = ["de", "en_us"]
phoneme_symbols = ["a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "r", "s", "t", "u", "v", "w", "x", "y", "z", "æ", "ç", "ð", "ø", "ŋ", "œ", "ɐ", "ɑ", "ɔ", "ə", "ɛ", "ɝ", "ɹ", "ɡ", "ɪ", "ʁ", "ʃ", "ʊ", "ʌ", "ʏ", "ʒ", "ʔ", "ˈ", "ˌ", "ː", "̃", "̍", "̥", "̩", "̯", "͡", "θ", "!", "?", ",", ".", "-", ":", ";", " ", '"', "'", "(", ")"]

phoneme_tokenizer = SequenceTokenizer(phoneme_symbols,
                                        languages=languages,
                                        lowercase=True,
                                        char_repeats=1,
                                        append_start_end=True)
tokens = phoneme_tokenizer("Hello, World!", "en_us")
tokens

# %%
phoneme_tokenizer.decode(tokens)

# %%
"ˈ" == "'"
# %%
