from lightning.pytorch.core import LightningModule
import torch

from models.config import (
    lang2id,
)
from models.vocoder.univnet import UnivNet

from .delightful_tts import DelightfulTTS


class DelightfulTTSModel(LightningModule):
    r"""DelightfulTTSModel is a PyTorch Lightning module that is used to use the DelightfulTTS inference.

    Args:
        delightfull_tts_ckpt (str): Path to the checkpoint of the DelightfulTTS model.
        lang (str): Language of the dataset.
    """

    def __init__(
        self,
        delightfull_tts_ckpt: str = "",
        lang: str = "en",
    ):
        super().__init__()

        self.lang = lang

        self.acoustic_model = DelightfulTTS.load_from_checkpoint(delightfull_tts_ckpt, lang=lang)
        self.acoustic_model.freeze()

        self.vocoder_module = UnivNet()
        self.vocoder_module.freeze()


    def forward(self, text: str, speaker_idx: torch.Tensor) -> torch.Tensor:
        r"""TTS inference.
        This code must be run only with the loaded weights from the checkpoint!

        Args:
            text (str): The input text.
            speaker_idx (torch.Tensor): The index of the speaker.
            lang (str): The language.

        Returns:
            torch.Tensor: The output of the AcousticModel.
        """
        normalized_text = self.normilize_text(text)
        _, phones = self.tokenizer(normalized_text)

        # Convert to tensor
        x = torch.tensor(
            phones, dtype=torch.int, device=speaker_idx.device,
        ).unsqueeze(0)

        speakers = speaker_idx.repeat(x.shape[1]).unsqueeze(0)

        langs = torch.tensor(
            [lang2id[self.lang]],
            dtype=torch.int,
            device=speaker_idx.device,
        ).repeat(x.shape[1]).unsqueeze(0)

        y_pred = self.acoustic_model(
            x=x,
            pitches_range=self.acoustic_model.pitches_stat,
            speakers=speakers,
            langs=langs,
        )

        return self.vocoder_module.forward(y_pred)
