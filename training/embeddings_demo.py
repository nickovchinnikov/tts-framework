# %%
import json

from dp.phonemizer import Phonemizer
from transformers import BertConfig, BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')

phonemizer = Phonemizer.from_checkpoint("../checkpoints/en_us_cmudict_ipa_forward.pt")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', vocab_file='../checkpoints/vocab2.txt')

configuration = BertConfig()
configuration

# %%
input_text = "Here is some text to encode"

tokenizer.encode(input_text, add_special_tokens=True)

# %%
phonemes_example = ['[SILENCE]', 'ð', 'ʌ', '[SILENCE]', 'd', 'eɪ', '[SILENCE]', 'æ', 'f', 't', 'ɜ˞', '[COMMA]', 'd', 'aɪ', 'æ', 'n', 'ʌ', '[SILENCE]', 'æ', 'n', 'd', '[SILENCE]', 'm', 'ɛ', 'ɹ', 'i', '[SILENCE]', 'k', 'w', 'ɪ', 't', 'ɪ', 'd', '[SILENCE]', 'ɪ', 't', '[SILENCE]', 'f', 'ɜ˞', '[SILENCE]', 'd', 'ɪ', 's', 't', 'ʌ', 'n', 't', '[SILENCE]', 'b', 'i', '[FULL STOP]']

embedding = tokenizer.encode(phonemes_example, add_special_tokens=True)
embedding

# %%
tokenizer.decode(embedding)

# %%
embedding2 = phonemizer.predictor.phoneme_tokenizer(phonemes_example, language="en_us")
embedding2

# %%
phonemizer.predictor.phoneme_tokenizer.decode(embedding2)

# %%
text = "".join(phonemes_example)
text

# %%
phone2ixd = json.load(open("../config/phone2idx.json"))
for key in phone2ixd.keys():
    print(key)

# %%