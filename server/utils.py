import io
import json

import re
from typing import Dict, List

import numpy as np
import pandas as pd
import soundfile as sf

from nltk.tokenize import sent_tokenize


def package_offsets(offsets: Dict) -> bytes:
    """Package wav and offsets with headers and frame separator"""

    batch = b"Content-Type: application/json\r\n\r\n"

    batch += bytes(json.dumps(offsets), "utf-8") + b"\r\n\r\n"

    batch += b"--frame\r\n"
    return batch


def returnAudioBuffer(
    audio_vector: np.ndarray,
    sample_rate: int,
    audio_format="mp3"
) -> io.BytesIO:
    """Return audio buffer"""
    
    buffer_ = io.BytesIO()
    audio_list = audio_vector.tolist()

    sf.write(buffer_, audio_list, sample_rate, format=audio_format)
    buffer_.seek(0)

    return buffer_


def audio_package(wav: np.ndarray, audio_format: str = "mp3") -> bytes:
    """Package audio"""
    sampling_rate = 24000

    batch = f"Content-Type: audio/{audio_format}\r\n\r\n".encode()

    audio_bytes = returnAudioBuffer(wav, sampling_rate, audio_format)

    batch += audio_bytes.read() + b"\r\n\r\n"

    batch += b"--frame\r\n"
    return batch


# TODO: Fix this, add direct speech detection
def sentences_split(text: str, max_symbols: int = 512):
    """Tokenize text into sentences no longer than `max_symbols`"""

    # sentences = sent_tokenize(text)
    # Split the text into sentences, treating direct speech as a single sentence
    # sentences = re.split(r'(?<!\"[A-Za-z0-9,])\.\s', text)
    # Find all instances of quoted speech
    # quoted_speech = re.findall(r'"[^"]*".?', text)

    # quoted_placeholders_dict = {}
    # # Replace quoted speech with placeholders
    # for i, quote in enumerate(quoted_speech):
    #     placeholder = f"__QUOTE{i}__"
    #     text = text.replace(quote, placeholder)
    #     quoted_placeholders_dict[placeholder] = quote

    # Split the text into sentences
    # sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    sentences = sent_tokenize(text)
    result, curr_split = [], ""

    for sentence in sentences:
        curr_split_length = len(curr_split)
        sentence_length = len(sentence)

        if curr_split_length + sentence_length <= max_symbols:
            curr_split += sentence if curr_split == "" else " " + sentence
        else:
            if curr_split_length == 0 or sentence_length > max_symbols or curr_split_length > max_symbols:
                raise ValueError(
                    f"Sentences in the text should be less than {max_symbols} symbols. "
                )

            result.append(curr_split)
            curr_split = sentence

    result.append(curr_split)
    return result
    
    # if not bool(quoted_placeholders_dict):
    #     return result

    # final_result = []
    # # Replace placeholders with quoted speech
    # for i, sentence in enumerate(result):
    #     placeholders = re.findall(r'__QUOTE\d+__', sentence)
    #     for placeholder in placeholders:
    #         fixed_sentence = sentence.replace(placeholder, quoted_placeholders_dict[placeholder])
    #         final_result.append(fixed_sentence)

    # return final_result


def speakers_info() -> Dict:
    speakers = json.load(
        open("config/speakers.json", "r")
    )
    return speakers


def generate_speakers() -> Dict:
    # Prepare the speaker information
    speakers_df = pd.read_csv("../config/speakers.tsv", sep="\t",)

    speakers_libriid_speakerid = json.load(
        open("../config/speaker_id_mapping_libri.json", "r")
    )

    speakers_dict_train_clean_100 = speakers_df[speakers_df["SUBSET"] == "train-clean-100"].to_dict('records')

    existed_speakers = {
        speakers_libriid_speakerid[str(row["READER"])]: {
            "id": speakers_libriid_speakerid[str(row["READER"])],
            "name": str(row["NAME"]),
            "gender": row["GENDER"],
        }
        for row in speakers_dict_train_clean_100
    }

    # Save the speakers information
    json.dump(existed_speakers, open("../config/speakers.json", "w"))

    return existed_speakers

