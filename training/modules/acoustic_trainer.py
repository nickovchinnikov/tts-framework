from typing import Any, Dict

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
from training.loss import LossesCriterion
from training.optimizer import ScheduledOptimFinetuning, ScheduledOptimPretraining


class AcousticTrainer(LightningModule):
    def __init__(self, fine_tuning: bool = False, lang: str = "en", root: str = "datasets_cache/LIBRITTS"):
        super().__init__()

        self.lang = lang
        self.root = root
        self.fine_tuning = fine_tuning

        self.train_config: AcousticTrainingConfig

        if self.fine_tuning:
            self.train_config = AcousticFinetuningConfig()
        else:
            self.train_config = AcousticPretrainingConfig()

        self.step = 0

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            n_speakers=5392,
        )

        self.loss = LossesCriterion()

    def weights_prepare(self, checkpoint_path: str):
        r"""
        Prepares the weights for the model. This is required for the model to be loaded from the checkpoint.
        """
        ckpt_acoustic = torch.load(checkpoint_path)

        for i in range(6):
            new_weights = torch.randn(384, 385) * 0.17
            existing_weights = ckpt_acoustic["gen"][
                f"decoder.layer_stack.{i}.conditioning.embedding_proj.weight"
            ]
            # Copy the existing weights into the new tensor
            new_weights[:, :-1] = existing_weights
            ckpt_acoustic["gen"][
                f"decoder.layer_stack.{i}.conditioning.embedding_proj.weight"
            ] = new_weights

        return ckpt_acoustic

    def forward(self, x: torch.Tensor):
        self.model
        return None

    # TODO: don't forget about torch.no_grad() !
    # default used by the Trainer
    # trainer = Trainer(inference_mode=True)
    # Use `torch.no_grad` instead
    # trainer = Trainer(inference_mode=False)
    def training_step(self, batch, batch_idx):
        # TODO: step is required for the future research, not sure if it's required...
        self.step += 1

        self.model.train()

        # TODO: check what inside the batch
        (
            ids,
            raw_texts,
            speakers,
            speaker_names,
            texts,
            src_lens,
            mels,
            pitches,
            mel_lens,
            langs,
            attn_priors,
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
            # TODO: fix pitches_range!
            pitches_range=(0.0, 1.0),
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
            step=self.step,
        )

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return super().on_load_checkpoint(checkpoint)

    def configure_optimizers(self):
        r"""
        Configures the optimizer used for training.
        Depending on the training mode, either the finetuning or the pretraining optimizer is used.

        Returns:
            ScheduledOptimFinetuning or ScheduledOptimPretraining: The optimizer.
        """
        if self.fine_tuning:
            scheduled_optim = ScheduledOptimFinetuning(
                parameters=self.model.parameters(),
                train_config=self.train_config,
                current_step=self.step,
            )
        else:
            scheduled_optim = ScheduledOptimPretraining(
                parameters=self.model.parameters(),
                train_config=self.train_config,
                current_step=self.step,
                model_config=self.model_config,
            )
        return scheduled_optim

    def train_dataloader(self):
        r"""
        Returns the training dataloader, that is using the LibriTTS dataset.

        Returns:
            DataLoader: The training dataloader.
        """

        dataset = LibriTTSDataset(
            root=self.root,
            batch_size=self.train_config.batch_size,
            lang=self.lang,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
        return loader
