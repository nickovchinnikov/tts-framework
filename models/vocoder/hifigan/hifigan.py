import itertools
from typing import List

from lightning.pytorch.core import LightningModule
import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    HifiGanConfig,
    HifiGanPretrainingConfig,
    PreprocessingConfig,
)
from training.datasets.hifi_gan_dataset import train_dataloader
from training.loss import (
    DiscriminatorLoss,
    FeatureMatchingLoss,
    GeneratorLoss,
)
from training.preprocess import TacotronSTFT

from .generator import Generator
from .mp_discriminator import MultiPeriodDiscriminator
from .ms_discriminator import MultiScaleDiscriminator


class HifiGan(LightningModule):
    r"""HifiGan module.

    This module contains the `Generator` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
        self,
        lang: str = "en",
        batch_size: int = 16,
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

        self.batch_size = batch_size
        self.sampling_rate = sampling_rate
        self.lang = lang

        self.preprocess_config = PreprocessingConfig(
            "multilingual",
            sampling_rate=sampling_rate,
        )
        self.train_config = HifiGanPretrainingConfig()

        self.generator = Generator(
            h=HifiGanConfig(),
            p=self.preprocess_config,
        )
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

        self.feature_loss = FeatureMatchingLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.generator_loss = GeneratorLoss()
        self.mae_loss = nn.L1Loss()

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

        # Mark TacotronSTFT as non-trainable
        for param in self.tacotronSTFT.parameters():
            param.requires_grad = False

        # Switch to manual optimization
        self.automatic_optimization = False

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
        audio_pred = self.generator.forward(mel)
        _, fake_mel = self.tacotronSTFT(audio_pred.squeeze(1))

        # Train discriminator
        opt_discriminator.zero_grad()
        mpd_score_real, mpd_score_gen, _, _ = self.mpd.forward(
            y=audio,
            y_hat=audio_pred.detach(),
        )
        loss_disc_mpd, _, _ = self.discriminator_loss.forward(
            disc_real_outputs=mpd_score_real,
            disc_generated_outputs=mpd_score_gen,
        )
        msd_score_real, msd_score_gen, _, _ = self.msd(
            y=audio,
            y_hat=audio_pred.detach(),
        )
        loss_disc_msd, _, _ = self.discriminator_loss(
            disc_real_outputs=msd_score_real,
            disc_generated_outputs=msd_score_gen,
        )
        loss_d = loss_disc_msd + loss_disc_mpd

        # Step for the discriminator
        self.manual_backward(loss_d, retain_graph=True)
        opt_discriminator.step()

        # Train generator
        opt_generator.zero_grad()
        loss_mel = self.mae_loss(fake_mel, mel)

        _, mpd_score_gen, fmap_mpd_real, fmap_mpd_gen = self.mpd.forward(
            y=audio,
            y_hat=audio_pred,
        )
        _, msd_score_gen, fmap_msd_real, fmap_msd_gen = self.msd.forward(
            y=audio,
            y_hat=audio_pred,
        )
        loss_fm_mpd = self.feature_loss.forward(
            fmap_r=fmap_mpd_real,
            fmap_g=fmap_mpd_gen,
        )
        loss_fm_msd = self.feature_loss.forward(
            fmap_r=fmap_msd_real,
            fmap_g=fmap_msd_gen,
        )
        loss_gen_mpd, _ = self.generator_loss.forward(disc_outputs=mpd_score_gen)
        loss_gen_msd, _ = self.generator_loss.forward(disc_outputs=msd_score_gen)
        loss_g = (
            loss_gen_msd
            + loss_gen_mpd
            + loss_fm_msd
            + loss_fm_mpd
            + loss_mel * self.train_config.l1_factor
        )

        # step for the generator
        self.manual_backward(loss_g, retain_graph=True)
        opt_generator.step()

        # Schedulers step
        sch_generator.step()
        sch_discriminator.step()

        # Gen losses
        self.log(
            "loss_gen_msd",
            loss_gen_msd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "loss_gen_mpd",
            loss_gen_mpd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "loss_fm_msd",
            loss_fm_msd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "loss_fm_mpd",
            loss_fm_mpd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "mel_loss",
            loss_mel,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        # Disc logs
        self.log(
            "loss_disc_msd",
            loss_disc_msd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "loss_disc_mpd",
            loss_disc_mpd,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "total_loss_disc",
            loss_d,
            sync_dist=True,
            batch_size=self.batch_size,
        )

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
            itertools.chain(self.msd.parameters(), self.mpd.parameters()),
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
