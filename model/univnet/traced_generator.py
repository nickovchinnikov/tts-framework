import torch
from torch.nn import Module

from model.helpers.tools import get_mask_from_lengths

from .generator import Generator


class TracedGenerator(Module):
    def __init__(
        self,
        generator: Generator,
        # example_inputs: Tuple[Any], example_inputs (Tuple[Any]): Example inputs to use for tracing.
    ):
        r"""A traced version of the Generator class that can be used for faster inference.

        Args:
            generator (Generator): The Generator instance to trace.
        """
        super().__init__()

        self.mel_mask_value: float = generator.mel_mask_value
        self.hop_length: int = generator.hop_length

        # TODO: https://github.com/Lightning-AI/lightning/issues/14036
        # Disable trace since model is non-deterministic
        # self.generator = torch.jit.trace(generator, example_inputs, check_trace=False)
        # self.generator = generator.to_torchscript(
        #     method="trace", example_inputs=example_inputs, check_trace=False
        # )
        self.generator = generator

    def forward(self, c: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        r"""Forward pass of the traced Generator.

        Args:
            c (torch.Tensor): The input mel-spectrogram tensor.
            mel_lens (torch.Tensor): The lengths of the input mel-spectrograms.

        Returns:
            torch.Tensor: The generated audio tensor.
        """
        mel_mask = get_mask_from_lengths(mel_lens).unsqueeze(1).to(c.device)
        c = c.masked_fill(mel_mask, self.mel_mask_value)
        zero = torch.full(
            (c.shape[0], c.shape[1], 10), self.mel_mask_value, device=c.device,
        )
        mel = torch.cat((c, zero), dim=2)
        audio = self.generator(mel)
        audio = audio[:, :, : -(self.hop_length * 10)]
        audio_mask = get_mask_from_lengths(mel_lens * 256).unsqueeze(1)
        return audio.masked_fill(audio_mask, 0.0)
