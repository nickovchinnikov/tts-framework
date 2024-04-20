from typing import List

from lightning.pytorch.core import LightningModule
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    HifiGanConfig,
    HifiGanPretrainingConfig,
    PreprocessingConfig,
)
from training.datasets.hifi_gan_dataset import train_dataloader
from training.loss import HifiLoss
from training.preprocess import TacotronSTFT

from .discriminator import Discriminator
from .generator import Generator


class HifiGan(LightningModule):
    r"""HifiGan module.

    This module contains the `Generator` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
        self,
        lang: str = "en",
        batch_size: int = 8,
        sampling_rate: int = 44100,
    ):
        r"""Initializes the `HifiGan`.

        Args:
            fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
            lang (str): Language of the dataset.
            batch_size (int): The batch size.
            sampling_rate (int): The sampling rate of the audio.
        """
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.lang = lang

        self.preprocess_config = PreprocessingConfig(
            "multilingual",
            sampling_rate=sampling_rate,
        )
        self.train_config = HifiGanPretrainingConfig()

        self.tacotronSTFT = TacotronSTFT(
            filter_length=self.preprocess_config.stft.filter_length,
            hop_length=self.preprocess_config.stft.hop_length,
            win_length=self.preprocess_config.stft.win_length,
            n_mel_channels=self.preprocess_config.stft.n_mel_channels,
            sampling_rate=self.preprocess_config.sampling_rate,
            mel_fmin=self.preprocess_config.stft.mel_fmin,
            mel_fmax=self.preprocess_config.stft.mel_fmax,
            center=False,
        )

        self.generator = Generator(
            h=HifiGanConfig(),
            p=self.preprocess_config,
        )
        self.discriminator = Discriminator()

        self.loss = HifiLoss()

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        r"""Performs a forward pass through the UnivNet model.

        Args:
            y_pred (torch.Tensor): The predicted mel spectrogram.

        Returns:
            torch.Tensor: The output of the UnivNet model.
        """
        wav_prediction = self.generator.forward(y_pred)

        return wav_prediction.squeeze()

    def training_step(self, batch: List, batch_idx: int):
        r"""Performs a training step for the model.

        Args:
            batch (Tuple[str, Tensor, Tensor]): The batch of data for training. Each item in the list is a tuple containing the ID of the item, the audio waveform, and the mel spectrogram.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the total loss for the generator and logs for tensorboard.
        """
        _, audio, mel = batch

        # Access your optimizers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        opt_generator: Optimizer = optimizers[0]  # type: ignore
        sch_generator: ExponentialLR = schedulers[0]  # type: ignore

        opt_discriminator: Optimizer = optimizers[1]  # type: ignore
        sch_discriminator: ExponentialLR = schedulers[1]  # type: ignore

        # Generate fake audio
        fake_audio = self.generator.forward(mel)

        _, fake_mel = self.tacotronSTFT(fake_audio.squeeze(1))

        # Discriminator
        mpd_res, msd_res = self.discriminator.forward(audio, fake_audio.detach())

        total_loss_disc, loss_disc_s, loss_disc_f = self.loss.desc_loss(
            audio,
            mpd_res,
            msd_res,
        )

        # Disc logs
        self.log(
            "total_loss_disc",
            total_loss_disc,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log("loss_disc_s", loss_disc_s, sync_dist=True, batch_size=self.batch_size)
        self.log("loss_disc_f", loss_disc_f, sync_dist=True, batch_size=self.batch_size)

        self.manual_backward(total_loss_disc, retain_graph=True)
        # step for the discriminator
        opt_discriminator.step()
        sch_discriminator.step()
        opt_discriminator.zero_grad()

        # Generator
        mpd_res, msd_res = self.discriminator.forward(audio, fake_audio)

        (
            total_loss_gen,
            loss_gen_f,
            loss_gen_s,
            loss_fm_s,
            loss_fm_f,
            stft_loss,
            loss_mel,
        ) = self.loss.gen_loss(
            audio,
            fake_audio,
            mel,
            fake_mel,
            mpd_res,
            msd_res,
        )

        self.log(
            "total_loss_gen",
            total_loss_gen,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        # Gen losses
        self.log("loss_gen_f", loss_gen_f, sync_dist=True, batch_size=self.batch_size)
        self.log("loss_gen_s", loss_gen_s, sync_dist=True, batch_size=self.batch_size)
        self.log("loss_fm_s", loss_fm_s, sync_dist=True, batch_size=self.batch_size)
        self.log("loss_fm_f", loss_fm_f, sync_dist=True, batch_size=self.batch_size)
        self.log("stft_loss", stft_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("loss_mel", loss_mel, sync_dist=True, batch_size=self.batch_size)

        # Perform manual optimization
        self.manual_backward(total_loss_gen, retain_graph=True)
        # step for the generator
        opt_generator.step()
        sch_generator.step()
        opt_generator.zero_grad()

    def configure_optimizers(self):
        r"""Configures the optimizers and learning rate schedulers for the `UnivNet` and `Discriminator` models.

        This method creates an `AdamW` optimizer and an `ExponentialLR` scheduler for each model.
        The learning rate, betas, and decay rate for the optimizers and schedulers are taken from the training configuration.

        Returns
            tuple: A tuple containing two dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.
        """
        optim_generator = AdamW(
            self.generator.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_generator = ExponentialLR(
            optim_generator,
            gamma=self.train_config.lr_decay,
            last_epoch=-1,
        )

        optim_discriminator = AdamW(
            self.discriminator.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_discriminator = ExponentialLR(
            optim_discriminator,
            gamma=self.train_config.lr_decay,
            last_epoch=-1,
        )

        return (
            {"optimizer": optim_generator, "lr_scheduler": scheduler_generator},
            {"optimizer": optim_discriminator, "lr_scheduler": scheduler_discriminator},
        )

    def train_dataloader(
        self,
        root: str = "datasets_cache",
        cache: bool = True,
        cache_dir: str = "/dev/shm",
    ) -> DataLoader:
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Args:
            root (str): The root directory of the dataset.
            cache (bool): Whether to cache the preprocessed data.
            cache_dir (str): The directory for the cache. Defaults to "/dev/shm".

        Returns:
            Tupple[DataLoader, DataLoader]: The training and validation dataloaders.
        """
        return train_dataloader(
            batch_size=self.batch_size,
            num_workers=self.preprocess_config.workers,
            root=root,
            cache=cache,
            cache_dir=cache_dir,
            lang=self.lang,
        )
