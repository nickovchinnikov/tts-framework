from lightning.pytorch.core import LightningModule
import torch
from torch import Tensor

from models.config import get_lang_map, lang2id
from models.tts.delightful_tts.delightful_tts_refined import DelightfulTTS
from models.vocoder.hifigan import HifiGan
from training.preprocess.normalize_text import NormalizeText

# Updated version of the tokenizer
from training.preprocess.tokenizer_ipa_espeak import TokenizerIpaEspeak as TokenizerIPA


class DelightfulHiFi(LightningModule):
    def __init__(
        self,
        delightful_checkpoint_path: str,
        hifi_checkpoint_path: str,
        lang: str = "en",
    ):
        super().__init__()

        lang_map = get_lang_map(lang)
        normilize_text_lang = lang_map.nemo

        self.normilize_text = NormalizeText(normilize_text_lang)
        self.tokenizer = TokenizerIPA(lang)

        self.delightful_tts = DelightfulTTS.load_from_checkpoint(
            delightful_checkpoint_path,
        )
        self.delightful_tts.freeze()
        self.hifi_gan = HifiGan.load_from_checkpoint(
            hifi_checkpoint_path,
        )
        self.hifi_gan.freeze()

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
