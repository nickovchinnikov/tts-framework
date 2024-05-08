from typing import List, Optional, Tuple

from lightning.pytorch.core import LightningModule
import torch
from torch.optim import AdamW, Optimizer, swa_utils
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    PreprocessingConfigUnivNet as PreprocessingConfig,
)
from models.config import (
    VocoderFinetuningConfig,
    VocoderModelConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
)
from models.helpers.dataloaders import train_dataloader
from training.loss import UnivnetLoss

from .discriminator import Discriminator
from .generator import Generator


class UnivNet(LightningModule):
    r"""Univnet module.

    This module contains the `Generator` and `Discriminator` models, and handles training and optimization.
    """

    def __init__(
        self,
        fine_tuning: bool = False,
        lang: str = "en",
        acc_grad_steps: int = 10,
        batch_size: int = 6,
        root: str = "datasets_cache/LIBRITTS",
        checkpoint_path_v1: Optional[str] = "checkpoints/vocoder_pretrained.pt",
    ):
        r"""Initializes the `VocoderModule`.

        Args:
            fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
            lang (str): Language of the dataset.
            acc_grad_steps (int): Accumulated gradient steps.
            batch_size (int): The batch size.
            root (str, optional): The root directory for the dataset. Defaults to "datasets_cache/LIBRITTS".
            checkpoint_path_v1 (str, optional): The path to the checkpoint for the model. If provided, the model weights will be loaded from this checkpoint. Defaults to None.
        """
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False
        self.acc_grad_steps = acc_grad_steps
        self.batch_size = batch_size

        self.lang = lang
        self.root = root

        model_config = VocoderModelConfig()
        preprocess_config = PreprocessingConfig("english_only")

        self.univnet = Generator(
            model_config=model_config,
            preprocess_config=preprocess_config,
        )
        self.discriminator = Discriminator(model_config=model_config)

        # Initialize SWA
        self.swa_averaged_univnet = swa_utils.AveragedModel(self.univnet)
        self.swa_averaged_discriminator = swa_utils.AveragedModel(self.discriminator)

        self.loss = UnivnetLoss()

        self.train_config: VoicoderTrainingConfig = (
            VocoderFinetuningConfig() if fine_tuning else VocoderPretrainingConfig()
        )

        # NOTE: this code is used only for the v0.1.0 checkpoint.
        # In the future, this code will be removed!
        self.checkpoint_path_v1 = checkpoint_path_v1
        if checkpoint_path_v1 is not None:
            generator, discriminator, _, _ = self.get_weights_v1(checkpoint_path_v1)
            self.univnet.load_state_dict(generator, strict=False)
            self.discriminator.load_state_dict(discriminator, strict=False)

    def get_weights_v1(self, checkpoint_path: str) -> Tuple[dict, dict, dict, dict]:
        r"""NOTE: this method is used only for the v0.1.0 checkpoint.
        Prepares the weights for the model.

        This is required for the model to be loaded from the checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint.

        Returns:
            Tuple[dict, dict, dict, dict]: The weights for the generator and discriminator.
        """
        ckpt_acoustic = torch.load(checkpoint_path)

        return (
            ckpt_acoustic["generator"],
            ckpt_acoustic["discriminator"],
            ckpt_acoustic["optim_g"],
            ckpt_acoustic["optim_d"],
        )

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        r"""Performs a forward pass through the UnivNet model.

        Args:
            y_pred (torch.Tensor): The predicted mel spectrogram.

        Returns:
            torch.Tensor: The output of the UnivNet model.
        """
        mel_lens = torch.tensor(
            [y_pred.shape[2]],
            dtype=torch.int32,
            device=y_pred.device,
        )

        wav_prediction = self.univnet.infer(y_pred, mel_lens)

        return wav_prediction[0, 0]

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
        fake_audio = self.univnet(mels)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

        (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
            esr_loss,
            snr_loss,
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
        self.log("esr_loss", esr_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("snr_loss", snr_loss, sync_dist=True, batch_size=self.batch_size)
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
            self.univnet.parameters(),
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

    def on_train_epoch_end(self):
        r"""Updates the averaged model after each optimizer step with SWA."""
        self.swa_averaged_univnet.update_parameters(self.univnet)
        self.swa_averaged_discriminator.update_parameters(self.discriminator)

    def on_train_end(self):
        # Update SWA model after training
        swa_utils.update_bn(self.train_dataloader(), self.swa_averaged_univnet)
        swa_utils.update_bn(self.train_dataloader(), self.swa_averaged_discriminator)

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
