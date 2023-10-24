from typing import List, Optional, Union

from pytorch_lightning.core import LightningModule
import torch
from torch.utils.data import DataLoader

from model.acoustic_model import AcousticModel
from model.config import (
    AcousticENModelConfig,
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
    AcousticTrainingConfig,
    PreprocessingConfig,
)
from model.helpers.tools import get_mask_from_lengths
from training.datasets import LibriTTSDataset
from training.loss import AcousticLoss
from training.optimizer import ScheduledOptimFinetuning, ScheduledOptimPretraining


class AcousticModule(LightningModule):
    r"""Trainer for the acoustic model.

    Args:
        fine_tuning (bool): Whether to use fine-tuning mode or not.
        lang (str): Language of the dataset.
        root (str): Root directory of the dataset.
        step (int): Current training step.
        n_speakers (int): Number of speakers in the dataset.
        checkpoint_path_v1 (Optional[str]): Path to the checkpoint to load the weights from. Note: this parameter is used only for the v0.1.0 checkpoint. In the future, this parameter will be removed!
    """

    def __init__(
            self,
            fine_tuning: bool = False,
            lang: str = "en",
            root: str = "datasets_cache/LIBRITTS",
            step: int = 0,
            n_speakers: int = 5392,
            checkpoint_path_v1: Optional[str] = None,
        ):
        super().__init__()

        self.lang = lang
        self.root = root
        self.fine_tuning = fine_tuning

        self.train_config: AcousticTrainingConfig

        if self.fine_tuning:
            self.train_config = AcousticFinetuningConfig()
        else:
            self.train_config = AcousticPretrainingConfig()

        self.step = step

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            # NOTE: this parameter may be hyperparameter that you can define based on the demands
            n_speakers=n_speakers,
        )

        self.loss = AcousticLoss()

        # NOTE: this code is used only for the v0.1.0 checkpoint.
        # In the future, this code will be removed!
        if checkpoint_path_v1 is not None:
            checkpoint = self._load_weights_v1(checkpoint_path_v1)
            self.model.load_state_dict(checkpoint, strict=False)

    def _load_weights_v1(self, checkpoint_path: str) -> dict:
        r"""NOTE: this method is used only for the v0.1.0 checkpoint.
        In the future, this method will be removed!
        Prepares the weights for the model.
        This is required for the model to be loaded from the checkpoint.
        """
        checkpoint = torch.load(checkpoint_path)

        # Fix the weights for the embedding projection
        for i in range(6):
            # 0.17 is the coff of standard deviation of the truncated normal distribution
            # Makes the range of distribution to be ~(-1, 1)
            new_weights = torch.randn(384, 385) * 0.17
            existing_weights = checkpoint["gen"][
                f"decoder.layer_stack.{i}.conditioning.embedding_proj.weight"
            ]
            # Copy the existing weights into the new tensor
            new_weights[:, :-1] = existing_weights
            checkpoint["gen"][
                f"decoder.layer_stack.{i}.conditioning.embedding_proj.weight"
            ] = new_weights

        return checkpoint["gen"]

    def forward(self, x: torch.Tensor):
        self.model

    # TODO: don't forget about torch.no_grad() !
    # default used by the Trainer
    # trainer = Trainer(inference_mode=True)
    # Use `torch.no_grad` instead
    # trainer = Trainer(inference_mode=False)
    def training_step(self, batch: List, batch_idx: int):
        (
            _,
            _,
            speakers,
            texts,
            src_lens,
            mels,
            pitches,
            pitches_stat,
            mel_lens,
            langs,
            attn_priors,
            _,
        ) = batch

        src_mask = get_mask_from_lengths(src_lens)
        mel_mask = get_mask_from_lengths(mel_lens)

        outputs = self.model.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=src_lens,
            mels=mels,
            mel_lens=mel_lens,
            pitches=pitches,
            pitches_range=pitches_stat,
            langs=langs,
            attn_priors=attn_priors,
        )

        y_pred = outputs["y_pred"]
        log_duration_prediction = outputs["log_duration_prediction"]
        p_prosody_ref = outputs["p_prosody_ref"]
        p_prosody_pred = outputs["p_prosody_pred"]
        pitch_prediction = outputs["pitch_prediction"]

        loss: float = self.loss.forward(
            src_mask=src_mask,
            src_lens=src_lens,
            mel_mask=mel_mask,
            mel_lens=mel_lens,
            mels=mels,
            y_pred=y_pred,
            log_duration_prediction=log_duration_prediction,
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_prediction=pitch_prediction,
            outputs=outputs,
            step=batch_idx + self.step,
        )

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self)-> Union[ScheduledOptimFinetuning, ScheduledOptimPretraining]:
        r"""Configures the optimizer used for training.
        Depending on the training mode, either the finetuning or the pretraining optimizer is used.

        Returns
            ScheduledOptimFinetuning or ScheduledOptimPretraining: The optimizer.
        """
        parameters = list(self.model.parameters())
        optimizer: Union[ScheduledOptimFinetuning, ScheduledOptimPretraining]

        if self.fine_tuning:
            optimizer = ScheduledOptimFinetuning(
                parameters=parameters,
                train_config=self.train_config,
                step=self.step,
            )
        else:
            optimizer = ScheduledOptimPretraining(
                parameters=parameters,
                train_config=self.train_config,
                model_config=self.model_config,
                step=self.step,
            )

        return optimizer

    def train_dataloader(self):
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Returns
            DataLoader: The training dataloader.
        """
        dataset = LibriTTSDataset(
            root=self.root,
            batch_size=self.train_config.batch_size,
            lang=self.lang,
        )
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
