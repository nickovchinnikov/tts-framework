# %%
import os
import sys

from dp.phonemizer import Phonemizer
from dp.preprocessing.text import SequenceTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

print(f"Current working directory: {os.getcwd()}")

checkpoint_ipa_forward = os.path.join(
    SCRIPT_DIR,
    "checkpoints",
    "en_us_cmudict_ipa_forward.pt",
)
checkpoint_ipa_forward

# %%
phonemizer = Phonemizer.from_checkpoint(checkpoint_ipa_forward)
phonemes = phonemizer("Phonemizing an English text is imposimpable!", lang="en_us")
phonemes

# %%
result = phonemizer.phonemise_list(
    "Phonemizing an English text is imposimpable!", lang="en_us"
)
result
# %%
result.text, result.phonemes[:2], result.split_text[:2], result.split_phonemes[
    :2
], result.predictions

# %%
# Tokenize the phomemes
result2 = phonemizer.predictor.phoneme_tokenizer(phonemes, language="en_us")
# As the result - tokenized phonemes
result2

# %% [markdown]
# Originally, used the following code to get the phonemes:
# word_tokenizer = WordTokenizer(lang=lang, remove_punct=False)
# sentence_tokenizer = SentenceTokenizer(lang=lang)
# acoustic_model = acoustic_model.to(device)
# start_time = time.time()
# if model_type == "Delighful_FreGANv1_v0.0" or model_type == "0.2.3":
#     waves = []
#     for sentence in sentence_tokenizer.tokenize(text):
#         symbol_ids = []
#         # sentence = text_normalizer(sentence)
#         for word in word_tokenizer.tokenize(sentence):
#             word = word.lower()
#             if word.strip() == "":
#                 continue
#             elif word in [".", "?", "!"]:
#                 symbol_ids.append(symbol2id[word])
#             elif word in [",", ";"]:
#                 symbol_ids.append(symbol2id["SILENCE"])
#             else:
#                 for phone in batched_predict(g2p, [word], [lang])[0]:
#                     symbol_ids.append(symbol2id[phone])
#                 symbol_ids.append(symbol2id["BLANK"])
# %%
