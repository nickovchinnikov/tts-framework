from typing import Callable, List, Tuple

from lightning.pytorch.core import LightningModule
from sklearn.model_selection import train_test_split
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader, SequentialSampler

from models.config import (
    AcousticENModelConfig,
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
    AcousticTrainingConfig,
    PreprocessingConfig,
    VocoderFinetuningConfig,
    # VocoderModelConfig,
    VocoderPretrainingConfig,
    VoicoderTrainingConfig,
    get_lang_map,
    lang2id,
)
from models.helpers.tools import get_mask_from_lengths
from models.vocoder.univnet import UnivNet
from training.datasets import LibriTTSDatasetAcoustic
from training.loss import FastSpeech2LossGen, Metrics
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
    """

    def __init__(
            self,
            fine_tuning: bool = False,
            lang: str = "en",
            n_speakers: int = 5392,
            batch_size: int = 6,
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

        self.train_config_vocoder: VoicoderTrainingConfig = \
        VocoderFinetuningConfig() \
        if fine_tuning \
        else VocoderPretrainingConfig()

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.acoustic_model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            # NOTE: this parameter may be hyperparameter that you can define based on the demands
            n_speakers=n_speakers,
        )

        # self.vocoder_config = VocoderModelConfig()

        # self.vocoder_module = UnivNet()
        # self.vocoder_module = self.vocoder_module.eval()
        # self.vocoder_module.freeze()

        # NOTE: in case of training from 0 bin_warmup should be True!
        self.loss_acoustic = FastSpeech2LossGen(fine_tuning=fine_tuning, bin_warmup=False)

        self.metrics = Metrics(lang)

        # Initialize pitches_stat with large/small values for min/max
        self.register_buffer("pitches_stat", torch.tensor([float("inf"), float("-inf")]))


    # def forward(self, text: str, speaker_idx: torch.Tensor, lang: str = "en") -> torch.Tensor:
    #     r"""Performs a forward pass through the AcousticModel.
    #     This code must be run only with the loaded weights from the checkpoint!

    #     Args:
    #         text (str): The input text.
    #         speaker_idx (torch.Tensor): The index of the speaker.
    #         lang (str): The language.

    #     Returns:
    #         torch.Tensor: The output of the AcousticModel.
    #     """
    #     normalized_text = self.normilize_text(text)
    #     _, phones = self.tokenizer(normalized_text)

    #     # Convert to tensor
    #     x = torch.tensor(
    #         phones, dtype=torch.int, device=speaker_idx.device,
    #     ).unsqueeze(0)

    #     speakers = speaker_idx.repeat(x.shape[1]).unsqueeze(0)

    #     langs = torch.tensor(
    #         [lang2id[lang]],
    #         dtype=torch.int,
    #         device=speaker_idx.device,
    #     ).repeat(x.shape[1]).unsqueeze(0)

    #     y_pred = self.acoustic_model(
    #         x=x,
    #         pitches_range=self.pitches_stat,
    #         speakers=speakers,
    #         langs=langs,
    #     )

    #     return self.vocoder_module.forward(y_pred)


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
            wavs,
            energies,
        ) = batch
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

        self.log("train_total_loss", total_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_mel_loss", mel_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_ssim_loss", ssim_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_duration_loss", duration_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_u_prosody_loss", u_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_p_prosody_loss", p_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_pitch_loss", pitch_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_ctc_loss", ctc_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_bin_loss", bin_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_energy_loss", energy_loss, sync_dist=True, batch_size=self.batch_size)

        return total_loss

        # # Access your optimizers
        # optimizers = self.optimizers()
        # schedulers = self.lr_schedulers()

        # ####################################
        # # Acoustic model manual optimizer ##
        # ####################################
        # opt_acoustic: Optimizer = optimizers[0] # type: ignore
        # sch_acoustic: LRScheduler = schedulers[0] # type: ignore

        # # Backward pass for the acoustic model
        # # NOTE: the loss is divided by the accumulated gradient steps
        # self.manual_backward(total_loss / self.acc_grad_steps)

        # # accumulate gradients of N batches
        # if (batch_idx + 1) % self.acc_grad_steps == 0:
        #     # clip gradients
        #     self.clip_gradients(opt_acoustic, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        #     # optimizer step
        #     opt_acoustic.step()
        #     # Scheduler step
        #     sch_acoustic.step()
        #     # zero the gradients
        #     opt_acoustic.zero_grad()


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
            speakers,
            texts,
            src_lens,
            mels,
            pitches,
            pitches_stat,
            mel_lens,
            langs,
            attn_priors,
            wavs,
            energies,
        ) = batch

        # Update pitches_stat (if needed)
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

        self.log("val_total_loss", total_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_mel_loss", mel_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_ssim_loss", ssim_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_duration_loss", duration_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_u_prosody_loss", u_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_p_prosody_loss", p_prosody_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_pitch_loss", pitch_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_ctc_loss", ctc_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_bin_loss", bin_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("val_energy_loss", energy_loss, sync_dist=True, batch_size=self.batch_size)


    def configure_optimizers(self):
        r"""Configures the optimizer used for training.
        Depending on the training mode, either the finetuning or the pretraining optimizer is used.

        Configures the optimizers and learning rate schedulers for the `AcousticModel`, `UnivNet` and `Discriminator` models.

        This method creates an `AdamW`, `Adam` optimizers and an `ExponentialLR`, `LambdaLR` schedulers.
        The learning rate, betas, and decay rate for the optimizers and schedulers are taken from the training configuration.

        The `Learning Rate Finder` is also configured, if self.learning_rate is not None.
        So, if the Learning Rate Finder is used, the optimizer is used self.learning_rate and the scheduler is not used.

        Returns
            tuple: A tuple containing three dictionaries. Each dictionary contains the optimizer and learning rate scheduler for one of the models.
        """
        parameters_acoustic = self.acoustic_model.parameters()

        # If the Learning Rate Finder is not used, the optimizer and the scheduler are used
        if self.fine_tuning:
            # Compute the gamma and initial learning rate based on the current step
            lr_decay = self.train_config_acoustic.optimizer_config.lr_decay
            default_lr = self.train_config_acoustic.optimizer_config.learning_rate

            init_lr = default_lr if self.trainer.global_step == 0 \
            else default_lr * (lr_decay ** self.trainer.global_step)

            optimizer = AdamW(
                parameters_acoustic,
                betas=self.train_config_acoustic.optimizer_config.betas,
                eps=self.train_config_acoustic.optimizer_config.eps,
                lr=init_lr,
            )

            scheduler = ExponentialLR(optimizer, gamma=lr_decay)

            return (
                {"optimizer": optimizer, "lr_scheduler": scheduler},
            )
        else:
            init_lr, lr_lambda = self.get_lr_lambda()

            optimizer = Adam(
                parameters_acoustic,
                betas=self.train_config_acoustic.optimizer_config.betas,
                eps=self.train_config_acoustic.optimizer_config.eps,
                lr=init_lr,
            )
            scheduler = LambdaLR(optimizer, lr_lambda)

            return (
                {"optimizer": optimizer, "lr_scheduler": scheduler},
            )


    def get_lr_lambda(self) -> Tuple[float, Callable[[int], float]]:
        r"""Returns the custom lambda function for the learning rate schedule.

        Returns
            function: The custom lambda function for the learning rate schedule.
        """
        init_lr = self.model_config.encoder.n_hidden ** -0.5

        def lr_lambda(step: int = self.trainer.global_step) -> float:
            r"""Computes the learning rate scale factor.

            Args:
                step (int): The current training step.

            Returns:
                float: The learning rate scale factor.
            """
            step = 1 if step == 0 else step

            warmup = self.train_config_acoustic.optimizer_config.warm_up_step
            anneal_steps = self.train_config_acoustic.optimizer_config.anneal_steps
            anneal_rate = self.train_config_acoustic.optimizer_config.anneal_rate

            lr_scale = min(
                step ** -0.5,
                step * warmup ** -1.5,
            )

            for s in anneal_steps:
                if step > s:
                    lr_scale *= anneal_rate

            return init_lr * lr_scale

        # Function that returns the learning rate scale factor
        return init_lr, lr_lambda


    def train_dataloader(
        self,
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
            batch_size=self.batch_size,
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
            batch_size=self.batch_size,
            # TODO: find the optimal num_workers
            num_workers=num_workers,
            sampler=val_sampler,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )

        return train_loader, val_loader
