from typing import Union

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

    def configure_optimizers(self):
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
