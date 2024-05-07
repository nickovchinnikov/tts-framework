from lightning.pytorch.core import LightningModule
from torch import Tensor

from models.config import PreprocessingConfigUnivNet as PreprocessingConfig
from models.tts.delightful_tts.delightful_tts import DelightfulTTS
from models.vocoder.univnet import UnivNet


class DelightfulUnivnet(LightningModule):
    def __init__(
        self,
        delightful_checkpoint_path: str,
        lang: str = "en",
        sampling_rate: int = 22050,
    ):
        super().__init__()

        self.preprocess_config = PreprocessingConfig(
            "multilingual",
            sampling_rate=sampling_rate,
        )

        self.delightful_tts = DelightfulTTS.load_from_checkpoint(
            delightful_checkpoint_path,
            # kwargs to be used in the model
            lang=lang,
            sampling_rate=sampling_rate,
            preprocess_config=self.preprocess_config,
        )
        self.delightful_tts.freeze()

        # Don't need to use separated checkpoint, prev checkpoint used
        self.univnet = UnivNet()
        self.univnet.freeze()

    def forward(
        self,
        text: str,
        speaker_idx: Tensor,
    ) -> Tensor:
        r"""Performs a forward pass through the AcousticModel.
        This code must be run only with the loaded weights from the checkpoint!

        Args:
            text (str): The input text.
            speaker_idx (Tensor): The index of the speaker.

        Returns:
            Tensor: The generated waveform with hifi-gan.
        """
        mel_pred = self.delightful_tts.forward(text, speaker_idx)

        wav = self.univnet.forward(mel_pred)

        return wav
