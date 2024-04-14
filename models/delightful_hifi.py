from lightning.pytorch.core import LightningModule
import torch
from torch import Tensor

from models.config import lang2id
from models.tts.delightful_tts.delightful_tts_refined import DelightfulTTS
from models.vocoder.hifigan import HifiGan


class DelightfulHiFi(LightningModule):
    def __init__(self, delightful_checkpoint_path: str, hifi_checkpoint_path: str):
        super().__init__()

        self.delightful_tts = DelightfulTTS.load_from_checkpoint(
            delightful_checkpoint_path,
        )
        self.hifi_gan = HifiGan.load_from_checkpoint(
            hifi_checkpoint_path,
        )

    def forward(
        self,
        text: str,
        speaker_idx: Tensor,
        lang: str = "en",
    ) -> Tensor:
        r"""Performs a forward pass through the AcousticModel.
        This code must be run only with the loaded weights from the checkpoint!

        Args:
            text (str): The input text.
            speaker_idx (Tensor): The index of the speaker.
            lang (str): The language.

        Returns:
            Tensor: The generated waveform with hifi-gan.
        """
        normalized_text = self.normilize_text(text)
        _, phones = self.tokenizer(normalized_text)

        # Convert to tensor
        x = torch.tensor(
            phones,
            dtype=torch.int,
            device=speaker_idx.device,
        ).unsqueeze(0)

        speakers = speaker_idx.repeat(x.shape[1]).unsqueeze(0)

        langs = (
            torch.tensor(
                [lang2id[lang]],
                dtype=torch.int,
                device=speaker_idx.device,
            )
            .repeat(x.shape[1])
            .unsqueeze(0)
        )

        mel_pred = self.delightful_tts.acoustic_model.forward(
            x=x,
            speakers=speakers,
            langs=langs,
        )

        wav = self.hifi_gan.generator.forward(mel_pred)

        return wav
