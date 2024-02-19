from typing import List, Optional, Tuple

from lightning.pytorch.core import LightningModule
from sklearn.model_selection import train_test_split
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, SequentialSampler

from models.config import (
    PreprocessingConfig,
    VocoderFinetuningConfig,
    VocoderModelConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
)
from training.datasets import LibriTTSDatasetAcoustic
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
            acc_grad_steps: int = 10,
            root: str = "datasets_cache/LIBRITTS",
            checkpoint_path_v1: Optional[str] = "checkpoints/vocoder_pretrained.pt",
        ):
        r"""Initializes the `VocoderModule`.

        Args:
            fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
            acc_grad_steps (int): Accumulated gradient steps.
            root (str, optional): The root directory for the dataset. Defaults to "datasets_cache/LIBRITTS".
            checkpoint_path_v1 (str, optional): The path to the checkpoint for the model. If provided, the model weights will be loaded from this checkpoint. Defaults to None.
        """
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False
        self.acc_grad_steps = acc_grad_steps

        self.root = root

        model_config = VocoderModelConfig()
        preprocess_config = PreprocessingConfig("english_only")

        self.univnet = Generator(
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
        mel_lens=torch.tensor(
            [y_pred.shape[2]], dtype=torch.int32, device=y_pred.device,
        )

        wav_prediction = self.univnet.infer(y_pred, mel_lens)

        return wav_prediction[0, 0]

    def training_step(self, batch: List, _: int, batch_idx: int):
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
        opt_univnet: Optimizer = optimizers[1] # type: ignore
        sch_univnet: ExponentialLR = schedulers[1] # type: ignore

        opt_discriminator: Optimizer = optimizers[2] # type: ignore
        sch_discriminator: ExponentialLR = schedulers[2] # type: ignore

        audio = wavs
        fake_audio = self.univnet(mels)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

        (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
        ) = self.loss_univnet.forward(
            audio,
            fake_audio,
            res_fake,
            period_fake,
            res_real,
            period_real,
        )

        self.log("total_loss_gen", total_loss_gen)
        self.log("total_loss_disc", total_loss_disc)
        self.log("stft_loss", stft_loss)
        self.log("score_loss", score_loss)

        # Perform manual optimization
        self.manual_backward(total_loss_gen / self.acc_grad_steps, retain_graph=True)
        self.manual_backward(total_loss_disc / self.acc_grad_steps, retain_graph=True)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.acc_grad_steps == 0:
            # clip gradients
            self.clip_gradients(opt_univnet, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            self.clip_gradients(opt_discriminator, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            # optimizer step
            opt_univnet.step()
            opt_discriminator.step()

            # Scheduler step
            sch_univnet.step()
            sch_discriminator.step()

            # zero the gradients
            opt_univnet.zero_grad()
            opt_discriminator.zero_grad()


    def validation_step(self, batch: List, batch_idx: int):
        r"""Performs a validation step for the model.

        Args:
        batch (List): The batch of data for training. The batch should contain:
            - ids: List of indexes.
            - raw_texts: Raw text inputs.
            - speakers: Speaker identities.
            - texts: Text inputs.
            - src_lens: Lengths of the source sequences.
            - mels: Mel spectrogram targets.
            - pitches: Pitch targets.
            - pitches_stat: Statistics of the pitches.
            - mel_lens: Lengths of the mel spectrograms.
            - langs: Language identities.
            - attn_priors: Prior attention weights.
            - wavs: Waveform targets.
            - energies: Energy targets.
        batch_idx (int): Index of the batch.

        Returns:
            - 'val_loss': The total loss for the training step.
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

        audio = wavs
        fake_audio = self.univnet(mels)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

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

        self.log("val_total_loss_gen", total_loss_gen)
        self.log("val_total_loss_disc", total_loss_disc)
        self.log("val_stft_loss", stft_loss)
        self.log("val_score_loss", score_loss)


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

    def train_dataloader(
        self,
        batch_size: int = 6,
        num_workers: int = 5,
        root: str = "datasets_cache/LIBRITTS",
        cache: bool = True,
        cache_dir: str = "datasets_cache",
        mem_cache: bool = False,
        url: str = "train-clean-360",
        validation_split: float = 0.05,  # Percentage of data to use for validation
    ) -> Tuple[DataLoader, DataLoader]:
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Args:
            batch_size (int): The batch size.
            num_workers (int): The number of workers.
            root (str): The root directory of the dataset.
            cache (bool): Whether to cache the preprocessed data.
            cache_dir (str): The directory for the cache.
            mem_cache (bool): Whether to use memory cache.
            url (str): The URL of the dataset.
            validation_split (float): The percentage of data to use for validation.

        Returns:
            Tupple[DataLoader, DataLoader]: The training and validation dataloaders.
        """
        dataset = LibriTTSDatasetAcoustic(
            root=root,
            lang=self.lang,
            cache=cache,
            cache_dir=cache_dir,
            mem_cache=mem_cache,
            url=url,
        )

        # Split dataset into train and validation
        train_indices, val_indices = train_test_split(
            list(range(len(dataset))),
            test_size=validation_split,
            random_state=42,
        )

        # Create Samplers
        train_sampler = SequentialSampler(train_indices)
        val_sampler = SequentialSampler(val_indices)

        # dataset = LibriTTSMMDatasetAcoustic("checkpoints/libri_preprocessed_data.pt")
        train_loader = DataLoader(
            dataset,
            # 4x80Gb max 10 sec audio
            # batch_size=20, # self.train_config.batch_size,
            # 4*80Gb max ~20.4 sec audio
            batch_size=batch_size,
            # TODO: find the optimal num_workers
            num_workers=num_workers,
            sampler=train_sampler,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        val_loader = DataLoader(
            dataset,
            # 4x80Gb max 10 sec audio
            # batch_size=20, # self.train_config.batch_size,
            # 4*80Gb max ~20.4 sec audio
            batch_size=batch_size,
            # TODO: find the optimal num_workers
            num_workers=num_workers,
            sampler=val_sampler,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        return train_loader, val_loader
