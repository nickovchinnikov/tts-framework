from typing import List

from lightning.pytorch.core import LightningModule
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    HifiGanConfig,
    PreprocessingConfig,
    VocoderFinetuningConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
)
from models.helpers.dataloaders import train_dataloader
from training.loss import HifiLoss

from .discriminator import Discriminator
from .generator import Generator


class HifiGan(LightningModule):
    r"""HifiGan module.

    This module contains the `Generator` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
        self,
        fine_tuning: bool = False,
        lang: str = "en",
        acc_grad_steps: int = 10,
        batch_size: int = 6,
        sampling_rate: int = 44100,
    ):
        r"""Initializes the `VocoderModule`.

        Args:
            fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
            lang (str): Language of the dataset.
            acc_grad_steps (int): Accumulated gradient steps.
            batch_size (int): The batch size.
            sampling_rate (int): The sampling rate of the audio.
        """
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False
        self.acc_grad_steps = acc_grad_steps
        self.batch_size = batch_size
        self.sampling_rate = sampling_rate

        self.lang = lang

        model_config = HifiGanConfig()
        preprocess_config = PreprocessingConfig(
            "english_only",
            sampling_rate=sampling_rate,
        )

        self.generator = Generator(
            h=model_config,
            p=preprocess_config,
        )
        self.discriminator = Discriminator()

        self.loss = HifiLoss()

        self.train_config: VoicoderTrainingConfig = (
            VocoderFinetuningConfig() if fine_tuning else VocoderPretrainingConfig()
        )

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
            batch (List): The batch of data for training. The batch should contain the mel spectrogram, its length, the audio, and the speaker ID.
            batch_idx (int): Index of the batch.

        Returns:
            dict: A dictionary containing the total loss for the generator and logs for tensorboard.
        """
        (
            _,
            _,
            _,
            _,
            _,
            mels,
            _,
            _,
            _,
            _,
            _,
            wavs,
            _,
        ) = batch

        # Access your optimizers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()
        opt_univnet: Optimizer = optimizers[0]  # type: ignore
        sch_univnet: ExponentialLR = schedulers[0]  # type: ignore

        opt_discriminator: Optimizer = optimizers[1]  # type: ignore
        sch_discriminator: ExponentialLR = schedulers[1]  # type: ignore

        audio = wavs
        fake_audio = self.generator(mels)

        (
            (msd_res_real, mpd_res_fake, _, _),
            (msd_period_real, mpd_period_fake, _, _),
        ) = self.discriminator.forward(
            audio,
            fake_audio.detach(),
        )

        (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
        ) = self.loss.forward(
            audio,
            fake_audio,
            res_fake,
            period_fake,
            res_real,
            period_real,
        )

        self.log(
            "total_loss_gen",
            total_loss_gen,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "total_loss_disc",
            total_loss_disc,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log("stft_loss", stft_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("score_loss", score_loss, sync_dist=True, batch_size=self.batch_size)

        # Perform manual optimization
        self.manual_backward(total_loss_gen / self.acc_grad_steps, retain_graph=True)
        self.manual_backward(total_loss_disc / self.acc_grad_steps, retain_graph=True)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.acc_grad_steps == 0:
            # clip gradients
            self.clip_gradients(
                opt_univnet,
                gradient_clip_val=0.5,
                gradient_clip_algorithm="norm",
            )
            self.clip_gradients(
                opt_discriminator,
                gradient_clip_val=0.5,
                gradient_clip_algorithm="norm",
            )

            # optimizer step
            opt_univnet.step()
            opt_discriminator.step()

            # Scheduler step
            sch_univnet.step()
            sch_discriminator.step()

            # zero the gradients
            opt_univnet.zero_grad()
            opt_discriminator.zero_grad()

    def configure_optimizers(self):
        r"""Configures the optimizers and learning rate schedulers for the `UnivNet` and `Discriminator` models.

        This method creates an `AdamW` optimizer and an `ExponentialLR` scheduler for each model.
        The learning rate, betas, and decay rate for the optimizers and schedulers are taken from the training configuration.

        Returns
            tuple: A tuple containing two dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.

        Examples
            ```python
            vocoder_module = VocoderModule()
            optimizers = vocoder_module.configure_optimizers()

            print(optimizers)
            (
                {"optimizer": <torch.optim.adamw.AdamW object at 0x7f8c0c0b3d90>, "lr_scheduler": <torch.optim.lr_scheduler.ExponentialLR object at 0x7f8c0c0b3e50>},
                {"optimizer": <torch.optim.adamw.AdamW object at 0x7f8c0c0b3f10>, "lr_scheduler": <torch.optim.lr_scheduler.ExponentialLR object at 0x7f8c0c0b3fd0>}
            )
            ```
        """
        optim_univnet = AdamW(
            self.generator.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_univnet = ExponentialLR(
            optim_univnet,
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

    def train_dataloader(
        self,
        num_workers: int = 5,
        root: str = "datasets_cache/LIBRITTS",
        cache: bool = True,
        cache_dir: str = "datasets_cache",
        mem_cache: bool = False,
        url: str = "train-clean-360",
    ) -> DataLoader:
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Args:
            num_workers (int): The number of workers.
            root (str): The root directory of the dataset.
            cache (bool): Whether to cache the preprocessed data.
            cache_dir (str): The directory for the cache.
            mem_cache (bool): Whether to use memory cache.
            url (str): The URL of the dataset.

        Returns:
            DataLoader: The training and validation dataloaders.
        """
        return train_dataloader(
            batch_size=self.batch_size,
            num_workers=num_workers,
            root=root,
            cache=cache,
            cache_dir=cache_dir,
            mem_cache=mem_cache,
            url=url,
            lang=self.lang,
        )
