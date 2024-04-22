from typing import List

from lightning.pytorch.core import LightningModule
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    AcousticFinetuningConfig,
    AcousticMultilingualModelConfig,
    AcousticPretrainingConfig,
    AcousticTrainingConfig,
    PreprocessingConfig,
    get_lang_map,
    lang2id,
)
from models.helpers.tools import get_mask_from_lengths
from training.datasets.hifi_libri_dataset import train_dataloader
from training.loss import FastSpeech2LossGen
from training.preprocess.normalize_text import NormalizeText

# Updated version of the tokenizer
from training.preprocess.tokenizer_ipa_espeak import TokenizerIpaEspeak as TokenizerIPA

from .acoustic_model import AcousticModel

MEL_SPEC_EVERY_N_STEPS = 1000
AUDIO_EVERY_N_STEPS = 100


class DelightfulTTS(LightningModule):
    r"""Trainer for the acoustic model.

    Args:
        fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
        lang (str): Language of the dataset.
        n_speakers (int): Number of speakers in the dataset.generation during training.
        batch_size (int): The batch size.
        sampling_rate (int): The sample rate of the audio.
    """

    def __init__(
        self,
        fine_tuning: bool = False,
        bin_warmup: bool = True,
        lang: str = "en",
        n_speakers: int = 5392,
        batch_size: int = 10,
        sampling_rate: int = 44100,
    ):
        super().__init__()

        self.lang = lang
        self.fine_tuning = fine_tuning
        self.batch_size = batch_size

        lang_map = get_lang_map(lang)
        normilize_text_lang = lang_map.nemo

        self.tokenizer = TokenizerIPA(lang)
        self.normilize_text = NormalizeText(normilize_text_lang)

        self.train_config_acoustic: AcousticTrainingConfig

        if self.fine_tuning:
            self.train_config_acoustic = AcousticFinetuningConfig()
        else:
            self.train_config_acoustic = AcousticPretrainingConfig()

        self.preprocess_config = PreprocessingConfig(
            "multilingual",
            sampling_rate=sampling_rate,
        )

        # NOTE: try Multilingual model config
        self.model_config = AcousticMultilingualModelConfig()

        # TODO: fix the arguments!
        self.acoustic_model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            # NOTE: this parameter may be hyperparameter that you can define based on the demands
            n_speakers=n_speakers,
        )

        # NOTE: in case of training from 0 bin_warmup should be True!
        self.loss_acoustic = FastSpeech2LossGen(bin_warmup=bin_warmup)

    def forward(
        self,
        text: str,
        speaker_idx: Tensor,
        lang: str = "en",
    ) -> Tensor:
        r"""Performs a forward pass through the AcousticModel.
        This code must be run only with the loaded weights from the checkpoint!

        Args:
            text (str): The input text.
            speaker_idx (Tensor): The index of the speaker.
            lang (str): The language.

        Returns:
            Tensor: The generated waveform with hifi-gan.
        """
        normalized_text = self.normilize_text(text)
        _, phones = self.tokenizer(normalized_text)

        # Convert to tensor
        x = torch.tensor(
            phones,
            dtype=torch.int,
            device=speaker_idx.device,
        ).unsqueeze(0)

        speakers = speaker_idx.repeat(x.shape[1]).unsqueeze(0)

        langs = (
            torch.tensor(
                [lang2id[lang]],
                dtype=torch.int,
                device=speaker_idx.device,
            )
            .repeat(x.shape[1])
            .unsqueeze(0)
        )

        mel_pred = self.acoustic_model.forward(
            x=x,
            speakers=speakers,
            langs=langs,
        )

        return mel_pred

        # wav = self.vocoder.forward(mel_pred)

        # return wav

    # TODO: don't forget about torch.no_grad() !
    # default used by the Trainer
    # trainer = Trainer(inference_mode=True)
    # Use `torch.no_grad` instead
    # trainer = Trainer(inference_mode=False)
    def training_step(self, batch: List, _: int):
        r"""Performs a training step for the model.

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
            - 'loss': The total loss for the training step.
        """
        (
            _,
            _,
            speakers,
            texts,
            src_lens,
            mels,
            pitches,
            _,
            mel_lens,
            langs,
            attn_priors,
            _,
            energies,
        ) = batch

        outputs = self.acoustic_model.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=src_lens,
            mels=mels,
            mel_lens=mel_lens,
            pitches=pitches,
            langs=langs,
            attn_priors=attn_priors,
            energies=energies,
        )

        y_pred = outputs["y_pred"]
        log_duration_prediction = outputs["log_duration_prediction"]
        p_prosody_ref = outputs["p_prosody_ref"]
        p_prosody_pred = outputs["p_prosody_pred"]
        pitch_prediction = outputs["pitch_prediction"]
        energy_pred = outputs["energy_pred"]
        energy_target = outputs["energy_target"]

        src_mask = get_mask_from_lengths(src_lens)
        mel_mask = get_mask_from_lengths(mel_lens)

        (
            total_loss,
            mel_loss,
            ssim_loss,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
            energy_loss,
        ) = self.loss_acoustic.forward(
            src_masks=src_mask,
            mel_masks=mel_mask,
            mel_targets=mels,
            mel_predictions=y_pred,
            log_duration_predictions=log_duration_prediction,
            u_prosody_ref=outputs["u_prosody_ref"],
            u_prosody_pred=outputs["u_prosody_pred"],
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_predictions=pitch_prediction,
            p_targets=outputs["pitch_target"],
            durations=outputs["attn_hard_dur"],
            attn_logprob=outputs["attn_logprob"],
            attn_soft=outputs["attn_soft"],
            attn_hard=outputs["attn_hard"],
            src_lens=src_lens,
            mel_lens=mel_lens,
            energy_pred=energy_pred,
            energy_target=energy_target,
            step=self.trainer.global_step,
        )

        self.log(
            "train_total_loss",
            total_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log("train_mel_loss", mel_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "train_ssim_loss",
            ssim_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_duration_loss",
            duration_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_u_prosody_loss",
            u_prosody_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_p_prosody_loss",
            p_prosody_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train_pitch_loss",
            pitch_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )
        self.log("train_ctc_loss", ctc_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_bin_loss", bin_loss, sync_dist=True, batch_size=self.batch_size)
        self.log(
            "train_energy_loss",
            energy_loss,
            sync_dist=True,
            batch_size=self.batch_size,
        )

        return total_loss

    def configure_optimizers(self):
        r"""Configures the optimizer used for training.

        Returns
            tuple: A tuple containing three dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.
        """
        lr_decay = self.train_config_acoustic.optimizer_config.lr_decay
        default_lr = self.train_config_acoustic.optimizer_config.learning_rate

        init_lr = (
            default_lr
            if self.trainer.global_step == 0
            else default_lr * (lr_decay**self.trainer.global_step)
        )

        optimizer_acoustic = AdamW(
            self.acoustic_model.parameters(),
            lr=init_lr,
            betas=self.train_config_acoustic.optimizer_config.betas,
            eps=self.train_config_acoustic.optimizer_config.eps,
            weight_decay=self.train_config_acoustic.optimizer_config.weight_decay,
        )

        scheduler_acoustic = ExponentialLR(optimizer_acoustic, gamma=lr_decay)

        return {
            "optimizer": optimizer_acoustic,
            "lr_scheduler": scheduler_acoustic,
        }

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
            sampling_rate=self.preprocess_config.sampling_rate,
            root=root,
            cache=cache,
            cache_dir=cache_dir,
            lang=self.lang,
        )
