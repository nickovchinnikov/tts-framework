from typing import List

from lightning.pytorch.core import LightningModule
import torch
from torch.optim import AdamW, Optimizer, swa_utils
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from models.config import (
    AcousticENModelConfig,
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
    AcousticTrainingConfig,
    PreprocessingConfig,
    VocoderFinetuningConfig,
    VocoderModelConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
    get_lang_map,
    lang2id,
)
from models.helpers.dataloaders import train_dataloader
from models.helpers.tools import get_mask_from_lengths

# Models
from models.tts.delightful_tts.acoustic_model import AcousticModel
from models.vocoder.univnet.discriminator import Discriminator
from models.vocoder.univnet.generator import Generator
from training.loss import FastSpeech2LossGen, UnivnetLoss
from training.preprocess.normalize_text import NormalizeText

# Updated version of the tokenizer
from training.preprocess.tokenizer_ipa_espeak import TokenizerIpaEspeak as TokenizerIPA


class DelightfulUnivnet(LightningModule):
    r"""Trainer for the acoustic model.

    Args:
        fine_tuning (bool, optional): Whether to use fine-tuning mode or not. Defaults to False.
        lang (str): Language of the dataset.
        n_speakers (int): Number of speakers in the dataset.generation during training.
        batch_size (int): The batch size.
        acc_grad_steps (int): The number of gradient accumulation steps.
        swa_steps (int): The number of steps for the SWA update.
    """

    def __init__(
            self,
            fine_tuning: bool = True,
            lang: str = "en",
            n_speakers: int = 5392,
            batch_size: int = 12,
            acc_grad_steps: int = 5,
            swa_steps: int = 1000,
        ):
        super().__init__()

        # Switch to manual optimization
        self.automatic_optimization = False
        self.acc_grad_steps = acc_grad_steps
        self.swa_steps = swa_steps

        self.lang = lang
        self.fine_tuning = fine_tuning
        self.batch_size = batch_size

        lang_map = get_lang_map(lang)
        normilize_text_lang = lang_map.nemo

        self.tokenizer = TokenizerIPA(lang)
        self.normilize_text = NormalizeText(normilize_text_lang)

        # Acoustic model
        self.train_config_acoustic: AcousticTrainingConfig

        if self.fine_tuning:
            self.train_config_acoustic = AcousticFinetuningConfig()
        else:
            self.train_config_acoustic = AcousticPretrainingConfig()

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config_acoustic = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.acoustic_model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config_acoustic,
            # NOTE: this parameter may be hyperparameter that you can define based on the demands
            n_speakers=n_speakers,
        )

        # Initialize SWA
        self.swa_averaged_acoustic = swa_utils.AveragedModel(self.acoustic_model)

        # NOTE: in case of training from 0 bin_warmup should be True!
        self.loss_acoustic = FastSpeech2LossGen(bin_warmup=False)

        # Initialize pitches_stat with large/small values for min/max
        self.register_buffer("pitches_stat", torch.tensor([float("inf"), float("-inf")]))

        # Vocoder models
        self.model_config_vocoder = VocoderModelConfig()

        self.train_config: VoicoderTrainingConfig = \
        VocoderFinetuningConfig() \
        if fine_tuning \
        else VocoderPretrainingConfig()

        self.univnet = Generator(
            model_config=self.model_config_vocoder,
            preprocess_config=self.preprocess_config,
        )
        self.swa_averaged_univnet = swa_utils.AveragedModel(self.univnet)

        self.discriminator = Discriminator(model_config=self.model_config_vocoder)
        self.swa_averaged_discriminator = swa_utils.AveragedModel(self.discriminator)

        self.loss_univnet = UnivnetLoss()

    def forward(self, text: str, speaker_idx: torch.Tensor, lang: str = "en") -> torch.Tensor:
        r"""Performs a forward pass through the AcousticModel.
        This code must be run only with the loaded weights from the checkpoint!

        Args:
            text (str): The input text.
            speaker_idx (torch.Tensor): The index of the speaker.
            lang (str): The language.

        Returns:
            torch.Tensor: The output of the AcousticModel.
        """
        normalized_text = self.normilize_text(text)
        _, phones = self.tokenizer(normalized_text)

        # Convert to tensor
        x = torch.tensor(
            phones, dtype=torch.int, device=speaker_idx.device,
        ).unsqueeze(0)

        speakers = speaker_idx.repeat(x.shape[1]).unsqueeze(0)

        langs = torch.tensor(
            [lang2id[lang]],
            dtype=torch.int,
            device=speaker_idx.device,
        ).repeat(x.shape[1]).unsqueeze(0)

        y_pred = self.acoustic_model.forward(
            x=x,
            pitches_range=self.pitches_stat,
            speakers=speakers,
            langs=langs,
        )

        mel_lens = torch.tensor(
            [y_pred.shape[2]], dtype=torch.int32, device=y_pred.device,
        )

        wav = self.univnet.infer(y_pred, mel_lens)

        return wav

    # TODO: don't forget about torch.no_grad() !
    # default used by the Trainer
    # trainer = Trainer(inference_mode=True)
    # Use `torch.no_grad` instead
    # trainer = Trainer(inference_mode=False)
    def training_step(self, batch: List, batch_idx: int):
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
            pitches_stat,
            mel_lens,
            langs,
            attn_priors,
            audio,
            energies,
        ) = batch

        #####################################
        ##    Acoustic model train step    ##
        #####################################

        # Update pitches_stat
        self.pitches_stat[0] = min(self.pitches_stat[0], pitches_stat[0])
        self.pitches_stat[1] = max(self.pitches_stat[1], pitches_stat[1])

        outputs = self.acoustic_model.forward_train(
            x=texts,
            speakers=speakers,
            src_lens=src_lens,
            mels=mels,
            mel_lens=mel_lens,
            pitches=pitches,
            pitches_range=pitches_stat,
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
            acc_total_loss,
            acc_mel_loss,
            acc_sc_mag_loss,
            acc_log_mag_loss,
            acc_ssim_loss,
            acc_duration_loss,
            acc_u_prosody_loss,
            acc_p_prosody_loss,
            acc_pitch_loss,
            acc_ctc_loss,
            acc_bin_loss,
            acc_energy_loss,
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

        self.log("acc_total_loss", acc_total_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_mel_loss", acc_mel_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_sc_mag_loss", acc_sc_mag_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_log_mag_loss", acc_log_mag_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_ssim_loss", acc_ssim_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_duration_loss", acc_duration_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_u_prosody_loss", acc_u_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_p_prosody_loss", acc_p_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_pitch_loss", acc_pitch_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_ctc_loss", acc_ctc_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_bin_loss", acc_bin_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("acc_energy_loss", acc_energy_loss, sync_dist=True, batch_size=self.batch_size)

        #####################################
        ##    Univnet model train step     ##
        #####################################
        fake_audio = self.univnet.forward(y_pred)

        res_fake, period_fake = self.discriminator(fake_audio.detach())
        res_real, period_real = self.discriminator(audio)

        (
            voc_total_loss_gen,
            voc_total_loss_disc,
            voc_stft_loss,
            voc_score_loss,
            voc_esr_loss,
            voc_snr_loss,
        ) = self.loss_univnet.forward(
            audio,
            fake_audio,
            res_fake,
            period_fake,
            res_real,
            period_real,
        )

        self.log("voc_total_loss_gen", voc_total_loss_gen, sync_dist=True, batch_size=self.batch_size)
        self.log("voc_total_loss_disc", voc_total_loss_disc, sync_dist=True, batch_size=self.batch_size)
        self.log("voc_stft_loss", voc_stft_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("voc_score_loss", voc_score_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("voc_esr_loss", voc_esr_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("voc_snr_loss", voc_snr_loss, sync_dist=True, batch_size=self.batch_size)

        # Manual optimizer
        # Access your optimizers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        ####################################
        # Acoustic model manual optimizer ##
        ####################################
        opt_acoustic: Optimizer = optimizers[0] # type: ignore
        sch_acoustic: ExponentialLR = schedulers[0] # type: ignore

        opt_univnet: Optimizer = optimizers[0] # type: ignore
        sch_univnet: ExponentialLR = schedulers[0] # type: ignore

        opt_discriminator: Optimizer = optimizers[1] # type: ignore
        sch_discriminator: ExponentialLR = schedulers[1] # type: ignore

        # Backward pass for the acoustic model
        # NOTE: the loss is divided by the accumulated gradient steps
        self.manual_backward(acc_total_loss / self.acc_grad_steps, retain_graph=True)

        # Perform manual optimization univnet
        self.manual_backward(voc_total_loss_gen / self.acc_grad_steps, retain_graph=True)
        self.manual_backward(voc_total_loss_disc / self.acc_grad_steps, retain_graph=True)

        # accumulate gradients of N batches
        if (batch_idx + 1) % self.acc_grad_steps == 0:
            # Acoustic model optimizer step
            # clip gradients
            self.clip_gradients(opt_acoustic, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

            # optimizer step
            opt_acoustic.step()
            # Scheduler step
            sch_acoustic.step()
            # zero the gradients
            opt_acoustic.zero_grad()

            # Univnet model optimizer step
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

        # Update SWA model every swa_steps
        if self.trainer.global_step % self.swa_steps == 0:
            self.swa_averaged_acoustic.update_parameters(self.acoustic_model)
            self.swa_averaged_univnet.update_parameters(self.univnet)
            self.swa_averaged_discriminator.update_parameters(self.discriminator)


    def on_train_epoch_end(self):
        r"""Updates the averaged model after each optimizer step with SWA."""
        self.swa_averaged_acoustic.update_parameters(self.acoustic_model)
        self.swa_averaged_univnet.update_parameters(self.univnet)
        self.swa_averaged_discriminator.update_parameters(self.discriminator)


    def configure_optimizers(self):
        r"""Configures the optimizer used for training.

        Returns
            tuple: A tuple containing three dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.
        """
        ####################################
        # Acoustic model optimizer config ##
        ####################################
        # Compute the gamma and initial learning rate based on the current step
        lr_decay = self.train_config_acoustic.optimizer_config.lr_decay
        default_lr = self.train_config_acoustic.optimizer_config.learning_rate

        init_lr = default_lr if self.trainer.global_step == 0 \
        else default_lr * (lr_decay ** self.trainer.global_step)

        optimizer_acoustic = AdamW(
            self.acoustic_model.parameters(),
            lr=init_lr,
            betas=self.train_config_acoustic.optimizer_config.betas,
            eps=self.train_config_acoustic.optimizer_config.eps,
            weight_decay=self.train_config_acoustic.optimizer_config.weight_decay,
        )

        scheduler_acoustic = ExponentialLR(optimizer_acoustic, gamma=lr_decay)

        ####################################
        # Univnet model optimizer config ##
        ####################################
        optim_univnet = AdamW(
            self.univnet.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_univnet = ExponentialLR(
            optim_univnet, gamma=self.train_config.lr_decay, last_epoch=-1,
        )

        ####################################
        # Discriminator optimizer config ##
        ####################################
        optim_discriminator = AdamW(
            self.discriminator.parameters(),
            self.train_config.learning_rate,
            betas=(self.train_config.adam_b1, self.train_config.adam_b2),
        )
        scheduler_discriminator = ExponentialLR(
            optim_discriminator, gamma=self.train_config.lr_decay, last_epoch=-1,
        )

        return (
            {"optimizer": optimizer_acoustic, "lr_scheduler": scheduler_acoustic},
            {"optimizer": optim_univnet, "lr_scheduler": scheduler_univnet},
            {"optimizer": optim_discriminator, "lr_scheduler": scheduler_discriminator},
        )


    def on_train_end(self):
        # Update SWA models after training
        swa_utils.update_bn(self.train_dataloader(), self.swa_averaged_acoustic)
        swa_utils.update_bn(self.train_dataloader(), self.swa_averaged_univnet)
        swa_utils.update_bn(self.train_dataloader(), self.swa_averaged_discriminator)


    def train_dataloader(
        self,
        num_workers: int = 5,
        root: str = "datasets_cache/LIBRITTS",
        cache: bool = True,
        cache_dir: str = "datasets_cache",
        mem_cache: bool = False,
        url: str = "train-clean-460",
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
            Tupple[DataLoader, DataLoader]: The training and validation dataloaders.
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
