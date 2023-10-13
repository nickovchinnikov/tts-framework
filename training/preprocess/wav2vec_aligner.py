from dataclasses import dataclass
import os
from typing import Tuple

from lightning.pytorch import LightningModule
import torch
import torchaudio
from transformers import (
    AutoConfig,
    AutoModelForCTC,
    AutoProcessor,
)


@dataclass
class Item:
    sent: str
    wav_path: str
    out_path: str

    
@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t{self.score:4.2f}\t{self.start*20:5d}\t{self.end*20:5d}"

    @property
    def length(self):
        return self.end - self.start


class Wav2VecAligner(LightningModule):
    def __init__(
            self,
            input_wavs_sr: int = 48_000,
            model_name: str = "facebook/wav2vec2-base-960h",
        ):
        r"""
        Initialize a new instance of the Wav2VecAligner class.

        Args:
            input_wavs_sr (int): The sample rate of the input wave files. Defaults to 48_000.
            model_name (str): The name of the pre-trained model to use. Defaults to "facebook/wav2vec2-base-960h".
        """

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


    def load_audio(self, wav_path: str) -> Tuple[torch.Tensor, int]:
        r"""
        Load an audio file from the specified path.

        Args:
            wav_path (str): The path to the audio file.

        Returns:
            Tuple[torch.Tensor, int]: A tuple containing the loaded audio data and the sample rate, or a FileNotFoundError if the file does not exist.
        """

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(wav_path, "Not found in wavs directory")

        speech_array, sampling_rate = torchaudio.load(wav_path)
        return speech_array, sampling_rate


    def align_single_sample(
        self,
        speech_array: torch.Tensor,
        text: str
    ) -> Tuple[torch.Tensor, list, str]:
        r"""
        Align a single sample of audio data with the corresponding text.

        Args:
            speech_array (torch.Tensor): The audio data.
            text (str): The corresponding text.

        Returns:
            Tuple[torch.Tensor, list, str]: A tuple containing the emissions, the tokens, and the transcript.
        """

        transcript = "|".join(text.split(" "))
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

        # Get the emission probability at frame level
        # Compute the probability in log-domain to avoid numerical instability
        # For this purpose, we normalize the emission with `torch.log_softmax()`
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = emissions[0]

        # get labels from vocab
        labels = ([""] + list(self.processor.tokenizer.get_vocab().keys()))[
            :-1
        ]  # logits don't align with the tokenizer's vocab

        dictionary = {c: i for i, c in enumerate(labels)}
        tokens = []
        for c in transcript:
            if c in dictionary:
                tokens.append(dictionary[c])

        return emissions, tokens, transcript
    

    def get_trellis(
        self,
        emission: torch.Tensor,
        tokens: list,
        blank_id: int = 0
    ) -> torch.Tensor:
        r"""
        Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
        that represents the probabilities of each source token being at a certain time step.

        Since we are looking for the most likely transitions, we take the more likely path for the value of $k_{(t+1,j+1)}$​, that is:

        $k_{t+1, j+1} = \max(k_{t, j} p_{t+1, c_{j+1}}, k_{t, j+1} p_{t+1, \text{repeat}})$

        Args:
            emission (torch.Tensor): The emission tensor.
            tokens (list): The list of tokens.
            blank_id (int): The ID of the blank token. Defaults to 0.

        Returns:
            torch.Tensor: The trellis matrix.
        """
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frames + 1, num_tokens + 1), -float("inf"))
        trellis[:, 0] = 0
        for t in range(num_frames):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis
    

    def backtrack(
        self,
        trellis: torch.Tensor,
        emission: torch.Tensor,
        tokens: list,
        blank_id: int=0
    ) -> list[Point]:
        r"""
        Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path.

        Args:
            trellis (torch.Tensor): The trellis matrix.
            emission (torch.Tensor): The emission tensor.
            tokens (list): The list of tokens.
            blank_id (int): The ID of the blank token. Defaults to 0.

        Returns:
            list[Point]: The optimal sequence alignment path.
        """
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = int(torch.argmax(trellis[:, j]).item())

        path: list[Point] = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]
    
    
    def merge_repeats(self, path: list[Point], transcript: str) -> list[Segment]:
        r"""
        Merge repeated tokens into a single segment.

        Args:
            path (list[Point]): The sequence alignment path.
            transcript (str): The transcript.

        Returns:
            list[Segment]: The list of segments.

        Note: this shouldn't affect repeated characters from the
        original sentences (e.g. `ll` in `hello`)
        """
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    transcript[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score,
                )
            )
            i1 = i2
        return segments
    

    # Merge words
    def merge_words(self, segments: list[Segment], separator="|") -> list[Segment]:
        r"""
        Merge words in the given path.

        Args:
            segments (list[Segment]): The list of segments.
            separator (str): The separator character. Defaults to "|".

        Returns:
            list[Segment]: The list of merged words.
        """
        words = []
        i1, i2 = 0, 0
        while i1 < len(segments):
            if i2 >= len(segments) or segments[i2].label == separator:
                if i1 != i2:
                    segs = segments[i1:i2]
                    word = "".join([seg.label for seg in segs])
                    score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                    words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
                i1 = i2 + 1
                i2 = i1
            else:
                i2 += 1
        return words


    def forward(self, wav_path: str, text: str) -> list[Segment]:
        r"""
        Perform the forward pass of the model, which involves loading the audio data, aligning the audio with the text,
        building the trellis, backtracking to find the optimal path, merging repeated tokens, and finally merging words.

        Args:
            wav_path (str): The path to the audio file.
            text (str): The corresponding text.

        Returns:
            list[Segment]: The list of segments representing the alignment of the audio data with the text.
        """

        audio_input, _ = self.load_audio(wav_path)
        
        emissions, tokens, transcript = self.align_single_sample(audio_input, text)

        trellis = self.get_trellis(emissions, tokens)

        path = self.backtrack(trellis, emissions, tokens)

        merged_path = self.merge_repeats(path, transcript)

        result = self.merge_words(merged_path)

        return result
