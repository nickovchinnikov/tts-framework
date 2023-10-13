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
            model_name: str = "facebook/wav2vec2-base-960h",
        ):
        r"""
        Initialize a new instance of the Wav2VecAligner class.

        Args:
            model_name (str): The name of the pre-trained model to use. Defaults to "facebook/wav2vec2-base-960h".
        """

        super().__init__()

        # Load the config
        self.config = AutoConfig.from_pretrained(model_name)

        self.model = AutoModelForCTC.from_pretrained(model_name)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model_name)

        # get labels from vocab
        self.labels = list(self.processor.tokenizer.get_vocab().keys())

        for i in range(len(self.labels)):
            if self.labels[i] == "[PAD]" or self.labels[i] == "<pad>":
                self.blank_id = i

        print("Blank Token id [PAD]/<pad>", self.blank_id)


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
        transcript = f"|{transcript}|"

        with torch.inference_mode():
            logits = self.model(speech_array).logits

        # Get the emission probability at frame level
        # Compute the probability in log-domain to avoid numerical instability
        # For this purpose, we normalize the emission with `torch.log_softmax()`
        emissions = torch.log_softmax(logits, dim=-1)
        emissions = emissions[0]

        dictionary = {c: i for i, c in enumerate(self.labels)}
        tokens = [dictionary[c] for c in transcript if c in dictionary]

        return emissions, tokens, transcript
    

    def get_trellis(
        self,
        emission: torch.Tensor,
        tokens: list,
    ) -> torch.Tensor:
        r"""
        Build a trellis matrix of shape (num_frames + 1, num_tokens + 1)
        that represents the probabilities of each source token being at a certain time step.

        Since we are looking for the most likely transitions, we take the more likely path for the value of $k_{(t+1,j+1)}$​, that is:

        $k_{t+1, j+1} = \max(k_{t, j} p_{t+1, c_{j+1}}, k_{t, j+1} p_{t+1, \text{repeat}})$

        Args:
            emission (torch.Tensor): The emission tensor.
            tokens (list): The list of tokens.

        Returns:
            torch.Tensor: The trellis matrix.
        """
        num_frames = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.

        trellis = torch.zeros((num_frames, num_tokens))
        trellis[1:, 0] = torch.cumsum(emission[1:, self.blank_id], 0)
        trellis[0, 1:] = -float("inf")
        trellis[-num_tokens + 1 :, 0] = float("inf")

        for t in range(num_frames - 1):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, self.blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens[1:]],
            )
        return trellis
    

    def backtrack(
        self,
        trellis: torch.Tensor,
        emission: torch.Tensor,
        tokens: list,
    ) -> list[Point]:
        r"""
        Walk backwards from the last (sentence_token, time_step) pair to build the optimal sequence alignment path.

        Args:
            trellis (torch.Tensor): The trellis matrix.
            emission (torch.Tensor): The emission tensor.
            tokens (list): The list of tokens.

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
        t, j = trellis.size(0) - 1, trellis.size(1) - 1

        path = [Point(j, t, emission[t, self.blank_id].exp().item())]
        while j > 0:
            # Should not happen but just in case
            assert t > 0

            # 1. Figure out if the current position was stay or change
            # Frame-wise score of stay vs change
            p_stay = emission[t - 1, self.blank_id]
            p_change = emission[t - 1, tokens[j]]

            # Context-aware score for stay vs change
            stayed = trellis[t - 1, j] + p_stay
            changed = trellis[t - 1, j - 1] + p_change

            # Update position
            t -= 1
            if changed > stayed:
                j -= 1

            # Store the path with frame-wise probability.
            prob = (p_change if changed > stayed else p_stay).exp().item()
            path.append(Point(j, t, prob))

        # Now j == 0, which means, it reached the SoS.
        # Fill up the rest for the sake of visualization
        while t > 0:
            prob = emission[t - 1, self.blank_id].exp().item()
            path.append(Point(j, t - 1, prob))
            t -= 1

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
    def merge_words(self, segments: list[Segment], separator: str = "|") -> list[Segment]:
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
    
    def save_segments(self, wav_path: str, text: str, save_dir: str):
        r"""
        Perform the forward pass of the model to get the segments and save each segment to a file.
        Used for debugging purposes.

        Args:
            wav_path (str): The path to the audio file.
            text (str): The corresponding text.
            save_dir (str): The directory where the audio files should be saved.

        Returns:
            None
        """
        word_segments = self.forward(wav_path, text)

        waveform, sampling_rate = self.load_audio(wav_path)

        emissions, tokens, _ = self.align_single_sample(waveform, text)

        trellis = self.get_trellis(emissions, tokens)

        ratio = waveform.size(1) / trellis.size(0)

        for i, word in enumerate(word_segments):
            x0 = int(ratio * word.start)
            x1 = int(ratio * word.end)

            print(f"{word.label} ({word.score:.2f}): {x0 / sampling_rate:.3f} - {x1 / sampling_rate:.3f} sec")

            segment_waveform = waveform[:, x0:x1]
            
            # Save the segment waveform to a file
            filename = f"{i}_{word.label}.wav"
            torchaudio.save(os.path.join(save_dir, filename), segment_waveform, sampling_rate)
