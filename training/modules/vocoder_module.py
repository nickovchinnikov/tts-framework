from typing import Any, List, Optional, Tuple

from pytorch_lightning.core import LightningModule
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from model.config import PreprocessingConfig, VocoderModelConfig, VoicoderTrainingConfig
from model.univnet import Discriminator, UnivNet
from training.datasets import LibriTTSDatasetVocoder
from training.loss import UnivnetLoss


class VocoderModule(LightningModule):
    r"""A PyTorch Lightning module for the Vocoder model.

    This module contains the `UnivNet` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
            self,
            train_config: VoicoderTrainingConfig,
            model_config: VocoderModelConfig,
            preprocess_config: PreprocessingConfig,
            root: str = "datasets_cache/LIBRITTS",
            checkpoint_path_v1: Optional[str] = None,
        ):
        r"""Initializes the `VocoderModule`.

        Args:
            train_config (VoicoderTrainingConfig): The training configuration.
            model_config (VocoderModelConfig): The model configuration.
            preprocess_config (PreprocessingConfig): The preprocessing configuration.
            root (str, optional): The root directory for the dataset. Defaults to "datasets_cache/LIBRITTS".
            checkpoint_path_v1 (Optional[str], optional): The path to the checkpoint for the model. If provided, the model weights will be loaded from this checkpoint. Defaults to None.
        """
        super().__init__()

        self.root = root

        self.univnet = UnivNet(
            model_config=model_config,
            preprocess_config=preprocess_config,
        )
        self.discriminator = Discriminator(model_config=model_config)

        self.loss = UnivnetLoss()

        self.train_config = train_config

        # NOTE: this code is used only for the v0.1.0 checkpoint.
        # In the future, this code will be removed!
        if checkpoint_path_v1 is not None:
            generator, discriminator = self.get_weights_v1(checkpoint_path_v1)
            self.univnet.load_state_dict(generator, strict=False)
            self.discriminator.load_state_dict(discriminator, strict=False)

    def get_weights_v1(self, checkpoint_path: str) -> Tuple[Any, Any]:
        r"""NOTE: this method is used only for the v0.1.0 checkpoint.
        Prepares the weights for the model.

        This is required for the model to be loaded from the checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint.

        Returns:
            Tuple[Any, Any]: The weights for the generator and discriminator.
        """
        ckpt_acoustic = torch.load(checkpoint_path)

        return ckpt_acoustic["generator"], ckpt_acoustic["discriminator"]

    def training_step(self, batch: List, batch_idx: int):
        r"""Performs a training step for the model.

        Args:
            batch (List): The batch of data for training. The batch should contain the mel spectrogram, its length, the audio, and the speaker ID.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the total loss for the generator and logs for tensorboard.
        """
        (
            mel,
            mel_len,
            audio,
            speaker_id,
        ) = batch

        fake_audio = self.univnet(mel)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

        (
            total_loss_gen,
            total_loss_disc,
            mel_loss,
            score_loss,
        ) = self.loss(
            audio,
            fake_audio,
            res_fake,
            period_fake,
            res_real,
            period_real,
        )

        tensorboard_logs = {
            "total_loss_gen": total_loss_gen,
            "total_loss_disc": total_loss_disc,
            "mel_loss": mel_loss,
            "score_loss": score_loss,
        }

        return {"loss": total_loss_gen, "log": tensorboard_logs}

    def configure_optimizers(self):
        r"""Configures the optimizers and learning rate schedulers for the `UnivNet` and `Discriminator` models.

        This method creates an `AdamW` optimizer and an `ExponentialLR` scheduler for each model.
        The learning rate, betas, and decay rate for the optimizers and schedulers are taken from the training configuration.

        Returns
            tuple: A tuple containing two dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.

        Examples
            >>> vocoder_module = VocoderModule()
            >>> optimizers = vocoder_module.configure_optimizers()
            >>> print(optimizers)
            (
                {"optimizer": <torch.optim.adamw.AdamW object at 0x7f8c0c0b3d90>, "lr_scheduler": <torch.optim.lr_scheduler.ExponentialLR object at 0x7f8c0c0b3e50>},
                {"optimizer": <torch.optim.adamw.AdamW object at 0x7f8c0c0b3f10>, "lr_scheduler": <torch.optim.lr_scheduler.ExponentialLR object at 0x7f8c0c0b3fd0>}
            )
        """
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

    def train_dataloader(self):
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Returns
            DataLoader: The training dataloader.
        """
        dataset = LibriTTSDatasetVocoder(
            root=self.root,
            batch_size=self.train_config.batch_size,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
