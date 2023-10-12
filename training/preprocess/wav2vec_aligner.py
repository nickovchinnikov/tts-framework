from dataclasses import dataclass
import os
from typing import Any, Tuple

from lightning.pytorch import LightningModule
import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)


@dataclass
class Item:
    sent: str
    wav_path: str
    out_path: str

class Wav2VecAligner(LightningModule):
    def __init__(
            self,
            input_wavs_sr: int = 48_000,
            model_name: str = "facebook/wav2vec2-base-960h",
        ):
        super().__init__()

        # Load the config
        self.config = AutoConfig.from_pretrained(model_name)

        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.resampler = torchaudio.transforms.Resample(input_wavs_sr, 16_000)

        blank_id = 0
        vocab = list(self.processor.tokenizer.get_vocab().keys())

        for i in range(len(vocab)):
            if vocab[i] == "[PAD]" or vocab[i] == "<pad>":
                blank_id = i

        print("Blank Token id [PAD]/<pad>", blank_id)
        self.blank_id = blank_id


    def load_audio(self, wav_path: str) -> Tuple[torch.Tensor, int, FileNotFoundError]:
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(wav_path, "Not found in wavs directory")

        speech_array, sampling_rate = torchaudio.load(wav_path)
        return speech_array, sampling_rate
    

    def text_to_transcript(self, text: str) -> str:
        result = "|".join(text.split(" "))
        return result


    def align_single_sample(
        self,
        speech_array: torch.Tensor,
        transcript: str
    ):
        speech_array = self.resampler(speech_array).squeeze()

        inputs = self.processor(
            speech_array,
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True
        )
        inputs = inputs.to(device=self.device)

        with torch.no_grad():
            logits = self.model(inputs.input_values).logits

        # get the emission probability at frame level
        emissions = torch.log_softmax(logits, dim=-1)
        emissions[0].cpu().detach()

        # get labels from vocab
        labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
            :-1
        ]  # logits don't align with the tokenizer's vocab

        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])

        return emissions, tokens

