# %%
import os
import sys

from dp.phonemizer import Phonemizer

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

# %%
libri_example = ['[SILENCE]', 'ð', 'ʌ', '[SILENCE]', 'd', 'eɪ', '[SILENCE]', 'æ', 'f', 't', 'ɜ˞', '[COMMA]', 'd', 'aɪ', 'æ', 'n', 'ʌ', '[SILENCE]', 'æ', 'n', 'd', '[SILENCE]', 'm', 'ɛ', 'ɹ', 'i', '[SILENCE]', 'k', 'w', 'ɪ', 't', 'ɪ', 'd', '[SILENCE]', 'ɪ', 't', '[SILENCE]', 'f', 'ɜ˞', '[SILENCE]', 'd', 'ɪ', 's', 't', 'ʌ', 'n', 't', '[SILENCE]', 'b', 'i', '[FULL STOP]']

result3 = phonemizer.predictor.phoneme_tokenizer(libri_example, language="en_us")
result3

# %%
# There is no tokenization for the special tokens! 
phonemizer.predictor.phoneme_tokenizer([';'], language="en_us")

# %%
# ruff: noqa: E402
import json

speaker2idx_path = os.path.join(
    SCRIPT_DIR,
    "config",
    "speaker2idx.json",
)

phone2idx_path = os.path.join(
    SCRIPT_DIR,
    "config",
    "phone2idx.json",
)

speaker2ixd = json.load(open("config/speaker2idx.json"))
phone2ixd = json.load(open("phone2idx.json"))

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
# ruff: noqa: E402
# from unidecode import unidecode

# unidecode('[’], [“], [”], [„], [–], [«], [»]')

# # %%
# # ruff: noqa: E402
# import nltk
# from nltk.tokenize import WordPunctTokenizer, sent_tokenize, word_tokenize

# # Download NLTK resources (only need to do this once)
# nltk.download('punkt')

# text="We're moving to L.A.!!! <<"
# Tokenizer=WordPunctTokenizer()
# print("We're moving to L.A.!!! :", Tokenizer.tokenize(text))

# # Sample text for tokenization
# text = "Hello....world"

# input_text = "Wow!!! This is amazing!!!"

# # Tokenize into words
# word_tokens = word_tokenize(text)
# print("Word Tokens:", word_tokens)



# # Custom tokenizer using regular expression
# pattern = r'\w+|[^\w\s]'  # Tokenize words or non-space non-word characters
# tokens = nltk.regexp_tokenize(input_text, r'\w+')

# print("Word Tokens input_text: ", word_tokenize(input_text))

# # Tokenize into sentences
# sentence_tokens = sent_tokenize(text)
# print("Sentence Tokens:", sentence_tokens)



# %%
import re

re.sub(r"(\.|\!|\?)\1{2,}", r"\1\1\1", 'Aha!!!!!! Are you there??? Got you....')

# re.sub(r"(\.|\!|\?)\1+", r"\1", 'Aha!!! Are you there??? Got you...')

# %%
# ruff: noqa: E402
import nemo_text_processing

# create text normalization instance that works on cased input
from nemo_text_processing.text_normalization.normalize import Normalizer

normalizer = Normalizer(input_case='cased', lang='en')

# run normalization on example string input
written = "We paid $123 for this desk."
normalized = normalizer.normalize(written, verbose=True, punct_post_process=True)
print(normalized)

# %%
text2 = "For example it normalizes 'medic' into 'm e d i c' or 'yeah' into 'y e a h'."

normalizer.normalize(text2, verbose=True, punct_post_process=True)

# %%
normalizer.normalize('medic, yeah', verbose=True, punct_post_process=True)

# %%
text3 = "St. Patrick's Day, spend $123 for this desk."

normalizer.normalize(text3, verbose=True, punct_post_process=True)

# %%
# Character is not normilized by the nemo_text_processing
text4 = "It’s a beautiful day… What????? I don't understand!!!!! «Do» ”this“ – and that."

normalizer.normalize(text4, verbose=True, punct_post_process=True)

# %%
text5 = "The alarm went off at 10:00a.m. \nI received $123"

normalizer.normalize(text5, verbose=True, punct_post_process=True)


# %%
text6 = "Mr. Smith paid $111 in U.S.A. on Dec. 17th. We paid $123 for this desk."

normalizer.normalize(text6, punct_post_process=True)


# %%

# %%
