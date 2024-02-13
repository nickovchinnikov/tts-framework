# %%
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# %%
# -*- coding: utf-8 -*-
"""
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
"""

def make_symbols(characters, phonemes, punctuations="!'(),-.:;? ", pad="_", eos="~", bos="^"):# pylint: disable=redefined-outer-name
    """Function to create symbols and phonemes"""
    _phonemes_sorted = sorted(list(phonemes))

    # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
    _arpabet = ["@" + s for s in _phonemes_sorted]

    # Export all symbols:
    _symbols = [pad, eos, bos] + list(characters) + _arpabet
    _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)

    return _symbols, _phonemes

_pad = "_"
_eos = "~"
_bos = "^"
_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!'(),-.:;? "
_punctuations = "!'(),-.:;? "
_phoneme_punctuations = ".!;:,?"

# Phonemes definition
_vowels = "iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ"
_non_pulmonic_consonants = "ʘɓǀɗǃʄǂɠǁʛ"
_pulmonic_consonants = "pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ"
_suprasegmentals = "ˈˌːˑ"
_other_symbols = "ʍwɥʜʢʡɕʑɺɧ"
_diacrilics = "ɚ˞ɫ"
_phonemes = _vowels + _non_pulmonic_consonants + _pulmonic_consonants + _suprasegmentals + _other_symbols + _diacrilics

symbols, phonemes = make_symbols(_characters, _phonemes, _punctuations, _pad, _eos, _bos)


# %%
import torch

from models.tts.tacotron.tacotron2 import Tacotron2

checkpoint_path="../checkpoints/best_model_mulspeak_vctk_ddc.pth.tar"

state = torch.load(checkpoint_path, map_location=torch.device("cpu"))

state

# %%

# self.load_state_dict(state["model"])

num_chars = 133
r = 2
ddc_r=4
decoder_in_features=768

tacotron2 = Tacotron2(
    num_chars, 1, r,
    decoder_in_features=decoder_in_features,
    ddc_r=ddc_r,
    double_decoder_consistency=True,
).load_state_dict(state["model"])
tacotron2

# %%

# %%

# %%

# %%
