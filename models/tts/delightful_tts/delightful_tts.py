from typing import Callable, List, Tuple

from lightning.pytorch.core import LightningModule
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader

from models.config import (
    AcousticENModelConfig,
    AcousticFinetuningConfig,
    AcousticPretrainingConfig,
    AcousticTrainingConfig,
    PreprocessingConfig,
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
        root (str): Root directory of the dataset.
        step (int): Current training step.
        n_speakers (int): Number of speakers in the dataset.generation during training.
    """

    def __init__(
            self,
            fine_tuning: bool = False,
            lang: str = "en",
            root: str = "datasets_cache/LIBRITTS",
            initial_step: int = 0,
            n_speakers: int = 5392,
        ):
        super().__init__()

        self.lang = lang
        self.root = root
        self.fine_tuning = fine_tuning

        lang_map = get_lang_map(lang)
        normilize_text_lang = lang_map.nemo

        self.tokenizer = TokenizerIPA(lang)
        self.normilize_text = NormalizeText(normilize_text_lang)

        self.train_config: AcousticTrainingConfig

        if self.fine_tuning:
            self.train_config = AcousticFinetuningConfig()
        else:
            self.train_config = AcousticPretrainingConfig()

        # TODO: check this argument!
        self.register_buffer("initial_step", torch.tensor(initial_step))

        self.preprocess_config = PreprocessingConfig("english_only")
        self.model_config = AcousticENModelConfig()

        # TODO: fix the arguments!
        self.acoustic_model = AcousticModel(
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            # NOTE: this parameter may be hyperparameter that you can define based on the demands
            n_speakers=n_speakers,
        )

        self.vocoder_module = UnivNet()
        self.vocoder_module = self.vocoder_module.eval()
        self.vocoder_module.freeze()

        self.loss = FastSpeech2LossGen(fine_tuning=fine_tuning)

        self.metrics = Metrics(lang)

        # Initialize pitches_stat with large/small values for min/max
        self.register_buffer("pitches_stat", torch.tensor([float("inf"), float("-inf")]))


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

        y_pred = self.acoustic_model(
            x=x,
            pitches_range=self.pitches_stat,
            speakers=speakers,
            langs=langs,
        )

        return self.vocoder_module.forward(y_pred)


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
        dict: A dictionary containing:
            - 'loss': The total loss for the training step.
            - 'log': A dictionary of tensorboard logs.
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
            _,
            energies,
        ) = batch

        # Update pitches_stat
        self.pitches_stat[0] = min(self.pitches_stat[0], pitches_stat[0])
        self.pitches_stat[1] = max(self.pitches_stat[1], pitches_stat[1])

        src_mask = get_mask_from_lengths(src_lens)
        mel_mask = get_mask_from_lengths(mel_lens)

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
        ) = self.loss(
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
            step=batch_idx + self.initial_step.item(),
        )

        tensorboard_logs = {
            "total_loss": total_loss.detach(),
            "mel_loss": mel_loss.detach(),
            "ssim_loss": ssim_loss.detach(),
            "duration_loss": duration_loss.detach(),
            "u_prosody_loss": u_prosody_loss.detach(),
            "p_prosody_loss": p_prosody_loss.detach(),
            "pitch_loss": pitch_loss.detach(),
            "ctc_loss": ctc_loss.detach(),
            "bin_loss": bin_loss.detach(),
            "energy_loss": energy_loss.detach(),
        }

        # Add the logs to the tensorboard
        if self.logger.experiment is not None: # type: ignore
            # Generate an audio ones in a while and save to tensorboard
            tensorboard = self.logger.experiment # type: ignore

            tensorboard.add_scalar("total_loss", total_loss, self.current_epoch)
            tensorboard.add_scalar("mel_loss", mel_loss, self.current_epoch)
            tensorboard.add_scalar("ssim_loss", ssim_loss, self.current_epoch)
            tensorboard.add_scalar("duration_loss", duration_loss, self.current_epoch)
            tensorboard.add_scalar("u_prosody_loss", u_prosody_loss, self.current_epoch)
            tensorboard.add_scalar("p_prosody_loss", p_prosody_loss, self.current_epoch)
            tensorboard.add_scalar("pitch_loss", pitch_loss, self.current_epoch)
            tensorboard.add_scalar("ctc_loss", ctc_loss, self.current_epoch)
            tensorboard.add_scalar("bin_loss", bin_loss, self.current_epoch)
            tensorboard.add_scalar("energy_loss", energy_loss, self.current_epoch)

        # TODO: check the initial_step, not sure that this's correct
        self.initial_step += torch.tensor(1)

        return {"loss": total_loss, "log": tensorboard_logs}


    def configure_optimizers(self):
        r"""Configures the optimizer used for training.
        Depending on the training mode, either the finetuning or the pretraining optimizer is used.

        The `Learning Rate Finder` is also configured, if self.learning_rate is not None.
        So, if the Learning Rate Finder is used, the optimizer is used self.learning_rate and the scheduler is not used.

        Returns
            dict: The dictionary containing the optimizer and the learning rate scheduler.
        """
        parameters = self.acoustic_model.parameters()

        # If the Learning Rate Finder is not used, the optimizer and the scheduler are used
        if self.fine_tuning:
            # Compute the gamma and initial learning rate based on the current step
            lr_decay = self.train_config.optimizer_config.lr_decay
            default_lr = self.train_config.optimizer_config.learning_rate

            init_lr = default_lr if self.initial_step.item() == 0 \
            else default_lr * (lr_decay ** self.initial_step.item())

            optimizer = AdamW(
                parameters,
                betas=self.train_config.optimizer_config.betas,
                eps=self.train_config.optimizer_config.eps,
                lr=init_lr,
            )

            scheduler = ExponentialLR(optimizer, gamma=lr_decay)

            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            init_lr, lr_lambda = self.get_lr_lambda()

            optimizer = Adam(
                parameters,
                betas=self.train_config.optimizer_config.betas,
                eps=self.train_config.optimizer_config.eps,
                lr=init_lr,
            )
            scheduler = LambdaLR(optimizer, lr_lambda)

            return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def get_lr_lambda(self) -> Tuple[float, Callable[[int], float]]:
        r"""Returns the custom lambda function for the learning rate schedule.

        Returns
            function: The custom lambda function for the learning rate schedule.
        """
        init_lr = self.model_config.encoder.n_hidden ** -0.5

        def lr_lambda(step: int = self.initial_step.item()) -> float:
            r"""Computes the learning rate scale factor.

            Args:
                step (int): The current training step.

            Returns:
                float: The learning rate scale factor.
            """
            step = 1 if step == 0 else step

            warmup = self.train_config.optimizer_config.warm_up_step
            anneal_steps = self.train_config.optimizer_config.anneal_steps
            anneal_rate = self.train_config.optimizer_config.anneal_rate

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
        batch_size: int = 7,
        num_workers: int = 5,
        cache: bool = True,
        mem_cache: bool = True,
    ) -> DataLoader:
        r"""Returns the training dataloader, that is using the LibriTTS dataset.

        Args:
            batch_size (int): The batch size.
            num_workers (int): The number of workers.
            cache (bool): Whether to cache the preprocessed data.
            mem_cache (bool): Whether to use memory cache.

        Returns:
            DataLoader: The training dataloader.
        """
        dataset = LibriTTSDatasetAcoustic(
            root=self.root,
            lang=self.lang,
            cache=cache,
            mem_cache=mem_cache,
        )

        # dataset = LibriTTSMMDatasetAcoustic("checkpoints/libri_preprocessed_data.pt")
        return DataLoader(
            dataset,
            # 4x80Gb max 10 sec audio
            # batch_size=20, # self.train_config.batch_size,
            # 4*80Gb max ~20.4 sec audio
            # batch_size=7,
            batch_size=batch_size,
            # TODO: find the optimal num_workers
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=dataset.collate_fn,
        )
