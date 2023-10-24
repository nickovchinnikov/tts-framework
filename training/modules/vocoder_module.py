from typing import Any, Optional, Tuple

from pytorch_lightning.core import LightningModule
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR

from model.config import PreprocessingConfig, VocoderModelConfig, VoicoderTrainingConfig
from model.univnet import Discriminator, UnivNet


class VocoderModule(LightningModule):
    def __init__(
            self,
            train_config: VoicoderTrainingConfig,
            model_config: VocoderModelConfig,
            preprocess_config: PreprocessingConfig,
            checkpoint_path_v1: Optional[str] = None,
        ):
        super().__init__()
        self.univnet = UnivNet(
            model_config=model_config,
            preprocess_config=preprocess_config,
        )
        self.discriminator = Discriminator(model_config=model_config)

        self.train_config = train_config

        if checkpoint_path_v1 is not None:
            generator, discriminator = self.get_weights_v1(checkpoint_path_v1)
            self.univnet.load_state_dict(generator, strict=False)
            self.discriminator.load_state_dict(discriminator, strict=False)

    def get_weights_v1(self, checkpoint_path: str) -> Tuple[Any, Any]:
        r"""Prepares the weights for the model.
        This is required for the model to be loaded from the checkpoint.
        """
        ckpt_acoustic = torch.load(checkpoint_path)

        return ckpt_acoustic["generator"], ckpt_acoustic["discriminator"]

    def train_dataloader(self):
        pass

    def configure_optimizers(self):
        optim_univnet = AdamW(
            self.univnet.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_univnet = ExponentialLR(
            optim_univnet, gamma=self.train_config.lr_decay, last_epoch=-1,
        )

        optim_discriminator = AdamW(
            self.discriminator.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_discriminator = ExponentialLR(
            optim_discriminator, gamma=self.train_config.lr_decay, last_epoch=-1,
        )

        return (
            {"optimizer": optim_univnet, "lr_scheduler": scheduler_univnet},
            {"optimizer": optim_discriminator, "lr_scheduler": scheduler_discriminator},
        )
