from typing import Union

from pytorch_lightning.core import LightningModule
import torch
from torch.utils.data import DataLoader

from model.acoustic_model import AcousticModel
from model.config import (
    AcousticENModelConfig,
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
)
from model.helpers.tools import get_mask_from_lengths
from training.dataset import LibriTTSDataset
from training.loss import LossesCriterion
from training.optimizer import ScheduledOptimFinetuning, ScheduledOptimPretraining


class AcousticModule(LightningModule):
    def __init__(self, fine_tuning: bool = False):
        super().__init__()

        self.fine_tuning = fine_tuning

        self.train_config: Union[AcousticFinetuningConfig, AcousticPretrainingConfig]

        if self.fine_tuning:
            self.train_config = AcousticFinetuningConfig()
        else:
            self.train_config = AcousticPretrainingConfig()

        self.step = 0

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.model = AcousticModel(
            data_path="model/config/",
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            fine_tuning=fine_tuning,
            n_speakers=5392,
            # Setup the device, because .to() under the hood of lightning is not working
            device=self.device,  # type: ignore
        )

        self.loss = LossesCriterion(device=self.device)  # type: ignore

        # print(self.model)

    def forward(self, x):
        self.model
        return True

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
        # TODO: fix filename, data_path, assets_path
        dataset = LibriTTSDataset(
            filename="val.txt",
            batch_size=1,
            sort=True,
            drop_last=False,
            data_path="data",
            assets_path="assets",
            is_eval=True,
        )
        loader = DataLoader(
            dataset,
            num_workers=4,
            batch_size=1,
            shuffle=True,
            collate_fn=dataset.collate_fn,
        )
        return loader
