from typing import Any, List, Optional, Tuple

from pytorch_lightning.core import LightningModule
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from model.config import (
    PreprocessingConfig,
    VocoderFinetuningConfig,
    VocoderModelConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
)
from model.univnet import Discriminator, UnivNet
from training.datasets import LibriTTSDatasetVocoder
from training.loss import UnivnetLoss


class VocoderModule(LightningModule):
    r"""A PyTorch Lightning module for the Vocoder model.

    This module contains the `UnivNet` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
            self,
            fine_tuning: bool = False,
            root: str = "datasets_cache/LIBRITTS",
            checkpoint_path_v1: Optional[str] = None,
        ):
        r"""Initializes the `VocoderModule`.

        Args:
            fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
            root (str, optional): The root directory for the dataset. Defaults to "datasets_cache/LIBRITTS".
            checkpoint_path_v1 (str, optional): The path to the checkpoint for the model. If provided, the model weights will be loaded from this checkpoint. Defaults to None.
        """
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False

        self.root = root

        model_config = VocoderModelConfig()
        preprocess_config = PreprocessingConfig("english_only")

        self.univnet = UnivNet(
            model_config=model_config,
            preprocess_config=preprocess_config,
        )
        self.discriminator = Discriminator(model_config=model_config)

        self.loss = UnivnetLoss()

        self.train_config: VoicoderTrainingConfig = \
        VocoderFinetuningConfig() \
        if fine_tuning \
        else VocoderPretrainingConfig()


        # NOTE: this code is used only for the v0.1.0 checkpoint.
        # In the future, this code will be removed!
        self.checkpoint_path_v1 = checkpoint_path_v1
        if checkpoint_path_v1 is not None:
            generator, discriminator, _, _ = self.get_weights_v1(checkpoint_path_v1)
            self.univnet.load_state_dict(generator, strict=False)
            self.discriminator.load_state_dict(discriminator, strict=False)

    def get_weights_v1(self, checkpoint_path: str) -> Tuple[Any, Any, Any, Any]:
        r"""NOTE: this method is used only for the v0.1.0 checkpoint.
        Prepares the weights for the model.

        This is required for the model to be loaded from the checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint.

        Returns:
            Tuple[Any, Any, Any, Any]: The weights for the generator and discriminator.
        """
        ckpt_acoustic = torch.load(checkpoint_path)

        return (
            ckpt_acoustic["generator"],
            ckpt_acoustic["discriminator"],
            ckpt_acoustic["optim_g"],
            ckpt_acoustic["optim_d"],
        )

    def training_step(self, batch: List, _: int):
        r"""Performs a training step for the model.

        Args:
            batch (List): The batch of data for training. The batch should contain the mel spectrogram, its length, the audio, and the speaker ID.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the total loss for the generator and logs for tensorboard.
        """
        # Access your optimizers
        optimizers: List[Optimizer] = self.optimizers() # type: ignore

        opt_univnet, opt_discriminator = optimizers

        (
            mel,
            _,
            audio,
            _,
        ) = batch

        audio = audio.unsqueeze(1)
        fake_audio = self.univnet(mel)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

        (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
        ) = self.loss(
            audio,
            fake_audio,
            res_fake,
            period_fake,
            res_real,
            period_real,
        )

        # Perform manual optimization
        # TODO: total_loss_gen shouldn't be float! Here is a bug!
        opt_univnet.zero_grad()
        self.manual_backward(total_loss_gen, retain_graph=True)
        opt_univnet.step()

        opt_discriminator.zero_grad()
        self.manual_backward(total_loss_disc)
        opt_discriminator.step()

        tensorboard_logs = {
            "total_loss_gen": total_loss_gen,
            "total_loss_disc": total_loss_disc,
            "mel_loss": stft_loss,
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

        # NOTE: this code is used only for the v0.1.0 checkpoint.
        # In the future, this code will be removed!
        if self.checkpoint_path_v1 is not None:
            _, _, optim_g, optim_d = self.get_weights_v1(self.checkpoint_path_v1)
            optim_univnet.load_state_dict(optim_g)
            optim_discriminator.load_state_dict(optim_d)

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
