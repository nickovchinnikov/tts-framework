from lightning.pytorch.core import LightningModule
from torch import Tensor

from models.config import PreprocessingConfigHifiGAN as PreprocessingConfig
from models.tts.delightful_tts.delightful_tts import DelightfulTTS
from models.vocoder.hifigan import HifiGan


class DelightfulHiFi(LightningModule):
    def __init__(
        self,
        delightful_checkpoint_path: str,
        hifi_checkpoint_path: str,
        lang: str = "en",
        sampling_rate: int = 44100,
    ):
        super().__init__()

        self.sampling_rate = sampling_rate

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

        self.hifi_gan = HifiGan.load_from_checkpoint(
            hifi_checkpoint_path,
        )
        self.hifi_gan.freeze()

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

        wav = self.hifi_gan.generator.forward(mel_pred)

        return wav
