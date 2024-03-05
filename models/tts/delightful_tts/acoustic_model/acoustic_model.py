from typing import Dict, Tuple, Union

import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from models.config import (
    SUPPORTED_LANGUAGES,
    AcousticModelConfigType,
    PreprocessingConfig,
    symbols,
)

# from models.enhancer.gaussian_diffusion import GaussianDiffusion
from models.helpers import (
    # pitch_phoneme_averaging,
    positional_encoding,
    tools,
)
from models.tts.delightful_tts.attention import Conformer
from models.tts.delightful_tts.constants import LEAKY_RELU_SLOPE
from models.tts.delightful_tts.reference_encoder import (
    PhonemeLevelProsodyEncoder,
    UtteranceLevelProsodyEncoder,
)

from .aligner import Aligner

# TODO: maybe I can use the energy adaptor for the next versions?
from .energy_adaptor import EnergyAdaptor
from .length_adaptor import LengthAdaptor
from .phoneme_prosody_predictor import PhonemeProsodyPredictor

# from .pitch_adaptor import PitchAdaptor
# from .pitch_adaptor2 import PitchAdaptor
from .pitch_adaptor_conv import PitchAdaptorConv

# from .post_net import PostNet


class AcousticModel(Module):
    r"""The DelightfulTTS AcousticModel class represents a PyTorch module for an acoustic model in text-to-speech (TTS).
    The acoustic model is responsible for predicting speech signals from phoneme sequences.

    The model comprises multiple sub-modules including encoder, decoder and various prosody encoders and predictors.
    Additionally, a pitch and length adaptor are instantiated.

    Args:
        preprocess_config (PreprocessingConfig): Object containing the configuration used for preprocessing the data
        model_config (AcousticModelConfigType): Configuration object containing various model parameters
        n_speakers (int): Total number of speakers in the dataset
        leaky_relu_slope (float, optional): Slope for the leaky relu. Defaults to LEAKY_RELU_SLOPE.

    Note:
        For more specific details on the implementation of sub-modules please refer to their individual respective modules.
    """

    def __init__(
        self,
        preprocess_config: PreprocessingConfig,
        model_config: AcousticModelConfigType,
        n_speakers: int,
        leaky_relu_slope: float = LEAKY_RELU_SLOPE,
    ):
        super().__init__()
        self.emb_dim = model_config.encoder.n_hidden

        self.encoder = Conformer(
            dim=model_config.encoder.n_hidden,
            n_layers=model_config.encoder.n_layers,
            n_heads=model_config.encoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim + model_config.lang_embed_dim,
            p_dropout=model_config.encoder.p_dropout,
            kernel_size_conv_mod=model_config.encoder.kernel_size_conv_mod,
            with_ff=model_config.encoder.with_ff,
        )

        self.pitch_adaptor_conv = PitchAdaptorConv(
            channels_in=model_config.encoder.n_hidden,
            channels_hidden=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            emb_kernel_size=model_config.variance_adaptor.emb_kernel_size,
            dropout=model_config.variance_adaptor.p_dropout,
            leaky_relu_slope=leaky_relu_slope,
        )

        self.energy_adaptor = EnergyAdaptor(
            channels_in=model_config.encoder.n_hidden,
            channels_hidden=model_config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=model_config.variance_adaptor.kernel_size,
            emb_kernel_size=model_config.variance_adaptor.emb_kernel_size,
            dropout=model_config.variance_adaptor.p_dropout,
            leaky_relu_slope=leaky_relu_slope,
        )

        self.length_regulator = LengthAdaptor(model_config)

        self.utterance_prosody_encoder = UtteranceLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )

        self.utterance_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config,
            phoneme_level=False,
        )

        self.phoneme_prosody_encoder = PhonemeLevelProsodyEncoder(
            preprocess_config,
            model_config,
        )

        self.phoneme_prosody_predictor = PhonemeProsodyPredictor(
            model_config=model_config,
            phoneme_level=True,
        )

        self.u_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_u,
            model_config.encoder.n_hidden,
        )

        self.u_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_u,
            elementwise_affine=False,
        )

        self.p_bottle_out = nn.Linear(
            model_config.reference_encoder.bottleneck_size_p,
            model_config.encoder.n_hidden,
        )

        self.p_norm = nn.LayerNorm(
            model_config.reference_encoder.bottleneck_size_p,
            elementwise_affine=False,
        )

        self.aligner = Aligner(
            d_enc_in=model_config.encoder.n_hidden,
            d_dec_in=preprocess_config.stft.n_mel_channels,
            d_hidden=model_config.encoder.n_hidden,
        )

        self.decoder = Conformer(
            dim=model_config.decoder.n_hidden,
            n_layers=model_config.decoder.n_layers,
            n_heads=model_config.decoder.n_heads,
            embedding_dim=model_config.speaker_embed_dim + model_config.lang_embed_dim,
            p_dropout=model_config.decoder.p_dropout,
            kernel_size_conv_mod=model_config.decoder.kernel_size_conv_mod,
            with_ff=model_config.decoder.with_ff,
        )

        self.src_word_emb = Parameter(
            tools.initialize_embeddings(
                (len(symbols), model_config.encoder.n_hidden),
            ),
        )

        self.to_mel = nn.Linear(
            model_config.decoder.n_hidden,
            preprocess_config.stft.n_mel_channels,
        )

        # NOTE: Instead of linear layer, we use 1D convolution
        # self.to_mel_conv = nn.Conv1d(
        #     model_config.decoder.n_hidden,
        #     preprocess_config.stft.n_mel_channels,
        #     kernel_size=7,
        #     padding=3,
        # )

        # Post net improve the quality of the mel spectrogram
        # It is a stack of 5 1D convolutional layers with 512 channels and kernel size 5
        # self.mel_post_net = PostNet(
        #     n_hidden=model_config.decoder.n_hidden,
        #     n_mel_channels=preprocess_config.stft.n_mel_channels,
        #     postnet_embedding_dim=model_config.postnet.postnet_embedding_dim,
        #     postnet_kernel_size=model_config.postnet.postnet_kernel_size,
        #     postnet_n_convolutions=model_config.postnet.postnet_n_convolutions,
        # )

        # self.diff_refiner = GaussianDiffusion(
        #     model_config.diffusion,
        # )

        # NOTE: here you can manage the speaker embeddings, can be used for the voice export ?
        # NOTE: flexibility of the model binded by the n_speaker parameter, maybe I can find another way?
        # NOTE: in LIBRITTS there are 2477 speakers, we can add more, just extend the speaker_embed matrix
        # Need to think about it more
        self.speaker_embed = Parameter(
            tools.initialize_embeddings(
                (n_speakers, model_config.speaker_embed_dim),
            ),
        )

        self.lang_embed = Parameter(
            tools.initialize_embeddings(
                (len(SUPPORTED_LANGUAGES), model_config.lang_embed_dim),
            ),
        )

    def get_embeddings(
        self,
        token_idx: torch.Tensor,
        speaker_idx: torch.Tensor,
        src_mask: torch.Tensor,
        lang_idx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Given the tokens, speakers, source mask, and language indices, compute
        the embeddings for tokens, speakers and languages and return the
        token_embeddings and combined speaker and language embeddings

        Args:
            token_idx (torch.Tensor): Tensor of token indices.
            speaker_idx (torch.Tensor): Tensor of speaker identities.
            src_mask (torch.Tensor): Mask tensor for source sequences.
            lang_idx (torch.Tensor): Tensor of language indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Token embeddings tensor,
            and combined speaker and language embeddings tensor.
        """
        token_embeddings = F.embedding(token_idx, self.src_word_emb)
        # NOTE: here you can manage the speaker embeddings, can be used for the voice export ?
        speaker_embeds = F.embedding(speaker_idx, self.speaker_embed)
        lang_embeds = F.embedding(lang_idx, self.lang_embed)

        # Merge the speaker and language embeddings
        embeddings = torch.cat([speaker_embeds, lang_embeds], dim=2)

        # Apply the mask to the embeddings and token embeddings
        embeddings = embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)

        return token_embeddings, embeddings

    def prepare_for_export(self) -> None:
        r"""Prepare the model for export.

        This method is called when the model is about to be exported, such as for deployment
        or serializing for later use. The method removes unnecessary components that are
        not needed during inference. Specifically, it removes the phoneme and utterance
        prosody encoders for this acoustic model. These components are typically used during
        training and are not needed when the model is used for making predictions.

        Returns
            None
        """
        del self.phoneme_prosody_encoder
        del self.utterance_prosody_encoder

    def freeze_exept_pitch_adaptor_conv(self) -> None:
        r"""Freeze all parameters except for the ones in the pitch adaptor."""
        for name, param in self.named_parameters():
            if "pitch_adaptor_conv" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_exept_post_net(self) -> None:
        r"""Freeze all parameters except for the ones in the post net."""
        for name, param in self.named_parameters():
            if "post_net" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def freeze_exept_conformer_blocks_ff(self) -> None:
        r"""Freeze all parameters except for the ones in the conformer blocks and feed-forward layers."""
        # Freeze all parameters
        for _, param in self.named_parameters():
                param.requires_grad = False

        # unfreeze the feed-forward layers in conformer blocks
        for _, module in self.named_modules():
            if isinstance(module, Conformer):
                for block in module.layer_stack:
                    for name, param in block.named_parameters():
                        if "ff" in name:
                            param.requires_grad = True

    # NOTE: freeze/unfreeze params changed, because of the conflict with the lightning module
    def freeze_params(self) -> None:
        r"""Freeze the trainable parameters in the model.

        By freezing, the parameters are no longer updated by gradient descent.
        This is typically done when you want to keep parts of your model fixed while training other parts.
        For this model, it freezes all parameters and then selectively unfreezes the
        speaker embeddings and the pitch adaptor's pitch embeddings to allow these components to update during training.

        Returns
            None
        """
        for par in self.parameters():
            par.requires_grad = False
        self.speaker_embed.requires_grad = True
        # NOTE: requires_grad prop
        # self.pitch_adaptor.pitch_embedding.embeddings.requires_grad = True

    # NOTE: freeze/unfreeze params changed, because of the conflict with the lightning module
    def unfreeze_params(self, freeze_text_embed: bool, freeze_lang_embed: bool) -> None:
        r"""Unfreeze the trainable parameters in the model, allowing them to be updated during training.

        This method is typically used to 'unfreeze' previously 'frozen' parameters, making them trainable again.
        For this model, it unfreezes all parameters and then selectively freezes the
        text embeddings and language embeddings, if required.

        Args:
            freeze_text_embed (bool): Flag to indicate if text embeddings should remain frozen.
            freeze_lang_embed (bool): Flag to indicate if language embeddings should remain frozen.

        Returns:
            None
        """
        # Iterate through all model parameters and make them trainable
        for par in self.parameters():
            par.requires_grad = True

        # If freeze_text_embed flag is True, keep the source word embeddings frozen
        if freeze_text_embed:
            # @fixed self.src_word_emb.parameters has no parameters() method!
            # for par in self.src_word_emb.parameters():
            self.src_word_emb.requires_grad = False

        # If freeze_lang_embed flag is True, keep the language embeddings frozen
        if freeze_lang_embed:
            self.lang_embed.requires_grad = False

    def average_utterance_prosody(
        self, u_prosody_pred: torch.Tensor, src_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Compute the average utterance prosody over the length of non-masked elements.

        This method averages the output of the utterance prosody predictor over
        the sequence lengths (non-masked elements). This function will return
        a tensor with the same first dimension but singleton trailing dimensions.

        Args:
            u_prosody_pred (torch.Tensor): Tensor containing the predicted utterance prosody of dimension (batch_size, T, n_features).
            src_mask (torch.Tensor): Tensor of dimension (batch_size, T) acting as a mask where masked entries are set to False.

        Returns:
            torch.Tensor: Tensor of dimension (batch_size, 1, n_features) containing average utterance prosody over non-masked sequence length.
        """
        # Compute the real sequence lengths by negating the mask and summing along the sequence dimension
        lengths = ((~src_mask) * 1.0).sum(1)

        # Compute the sum of u_prosody_pred across the sequence length dimension,
        #  then divide by the sequence lengths tensor to calculate the average.
        #  This performs a broadcasting operation to account for the third dimension (n_features).
        # Return the averaged prosody prediction
        return u_prosody_pred.sum(1, keepdim=True) / lengths.view(-1, 1, 1)

    def forward_train(
        self,
        x: torch.Tensor,
        speakers: torch.Tensor,
        src_lens: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        pitches: torch.Tensor,
        pitches_range: Tuple[float, float],
        langs: torch.Tensor,
        attn_priors: Union[torch.Tensor, None],
        energies: torch.Tensor,
        use_ground_truth: bool = True,
    ) -> Dict[str, torch.Tensor]:
        r"""Forward pass during training phase.

        For a given phoneme sequence, speaker identities, sequence lengths, mels,
        mel lengths, pitches, language, and attention priors, the forward pass
        processes these inputs through the defined architecture.

        Args:
            x (torch.Tensor): Tensor of phoneme sequence.
            speakers (torch.Tensor): Tensor of speaker identities.
            src_lens (torch.Tensor): Long tensor representing the lengths of source sequences.
            mels (torch.Tensor): Tensor of mel spectrograms.
            mel_lens (torch.Tensor): Long tensor representing the lengths of mel sequences.
            pitches (torch.Tensor): Tensor of pitch values.
            pitches_range (Tuple[float, float]): The pitch min/max range.
            langs (torch.Tensor): Tensor of language identities.
            attn_priors (torch.Tensor): Prior attention values.
            energies (torch.Tensor): Tensor of energy values.
            use_ground_truth (bool, optional): Flag indicating whether ground truth should be used.
                                               Defaults to True.

        Returns:
            Dict[str, torch.Tensor]: Returns the prediction outputs as a dictionary.
        """
        # Generate masks for padding positions in the source sequences and mel sequences
        src_mask = tools.get_mask_from_lengths(src_lens)
        mel_mask = tools.get_mask_from_lengths(mel_lens)

        x, embeddings = self.get_embeddings(
            token_idx=x, speaker_idx=speakers, src_mask=src_mask, lang_idx=langs,
        )

        encoding = positional_encoding(
            self.emb_dim,
            max(x.shape[1], int(mel_lens.max().item())),
        )
        x = x.to(src_mask.device)
        encoding = encoding.to(src_mask.device)
        embeddings = embeddings.to(src_mask.device)

        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        u_prosody_ref = self.u_norm(
            self.utterance_prosody_encoder(mels=mels, mel_lens=mel_lens),
        )
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=x, mask=src_mask),
                src_mask=src_mask,
            ),
        )

        p_prosody_ref = self.p_norm(
            self.phoneme_prosody_encoder(
                x=x, src_mask=src_mask, mels=mels, mel_lens=mel_lens, encoding=encoding,
            ),
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(
                x=x,
                mask=src_mask,
            ),
        )
        if use_ground_truth:
            x = x + self.u_bottle_out(u_prosody_ref)
            x = x + self.p_bottle_out(p_prosody_ref)
        else:
            x = x + self.u_bottle_out(u_prosody_pred)
            x = x + self.p_bottle_out(p_prosody_pred)
        x_res = x
        attn_logprob, attn_soft, attn_hard, attn_hard_dur = self.aligner(
            enc_in=x_res.permute((0, 2, 1)),
            dec_in=mels,
            enc_len=src_lens,
            dec_len=mel_lens,
            enc_mask=src_mask,
            attn_prior=attn_priors,
        )

        # NOTE: changed the pitch adaptor to the new one
        # pitches = pitch_phoneme_averaging(
        #     durations=attn_hard_dur, pitches=pitches, max_phoneme_len=x.shape[1],
        # )
        # x, pitch_prediction, _, _ = self.pitch_adaptor.add_pitch_train(
        #     x=x,
        #     pitch_range=pitches_range,
        #     pitch_target=pitches,
        #     src_mask=src_mask,
        #     use_ground_truth=use_ground_truth,
        # )

        attn_hard_dur = attn_hard_dur.to(src_mask.device)

        x, pitch_prediction, avg_pitch_target = self.pitch_adaptor_conv.add_pitch_embedding_train(
            x=x,
            target=pitches,
            dr=attn_hard_dur,
            mask=src_mask,
        )

        energies = energies.to(src_mask.device)

        # NOTE: add the energy embedding to the encoder output
        x, energy_pred, avg_energy_target = self.energy_adaptor.add_energy_embedding_train(
            x=x,
            target=energies,
            dr=attn_hard_dur,
            mask=src_mask,
        )

        """assert torch.equal(attn_hard_dur.sum(1).long(), mel_lens), (
            attn_hard_dur.sum(1),
            mel_lens,
        )"""
        cond, log_duration_prediction, embeddings = self.length_regulator.upsample_train(
            x=x,
            x_res=x_res,
            duration_target=attn_hard_dur,
            src_mask=src_mask,
            embeddings=embeddings,
        )

        # Decode the encoder output to pred mel spectrogram
        decoder_output = self.decoder(cond, mel_mask, embeddings=embeddings, encoding=encoding)

        x = self.to_mel(decoder_output)
        x = x.permute((0, 2, 1))

        # decoder_output = decoder_output.permute((1, 2, 0))
        # x = self.to_mel_conv(decoder_output)

        # Post net synthesis of the mel spectrogram
        # x_postnet = self.mel_post_net.forward(decoder_output.permute((1, 2, 0)))
        # x_postnet = x_postnet.permute((0, 2, 1))
        # x_postnet, _, _, _, _ = self.diff_refiner.forward(
        #     mel=mels,
        #     cond=cond,
        #     spk_emb=embeddings,
        #     mel_mask=mel_mask,
        #     coarse_mel=x,
        # )
        # x_postnet = x_postnet.permute((0, 2, 1))

        return {
            "y_pred": x,
            # "y_postnet": x_postnet,
            "pitch_prediction": pitch_prediction,
            "pitch_target": avg_pitch_target,
            "energy_pred": energy_pred,
            "energy_target": avg_energy_target,
            "log_duration_prediction": log_duration_prediction,
            "u_prosody_pred": u_prosody_pred,
            "u_prosody_ref": u_prosody_ref,
            "p_prosody_pred": p_prosody_pred,
            "p_prosody_ref": p_prosody_ref,
            "attn_logprob": attn_logprob,
            "attn_soft": attn_soft,
            "attn_hard": attn_hard,
            "attn_hard_dur": attn_hard_dur,
        }

    def forward(
        self,
        x: torch.Tensor,
        pitches_range: Tuple[float, float],
        speakers: torch.Tensor,
        langs: torch.Tensor,
        p_control: float = 1.0,
        d_control: float = 1.0,
    ) -> torch.Tensor:
        r"""Forward pass during model inference.

        The forward pass receives phoneme sequence, speaker identities, languages, pitch control and
        duration control, conducts a series of operations on these inputs and returns the predicted mel
        spectrogram.

        Args:
            x (torch.Tensor): Tensor of phoneme sequences.
            pitches_range (Tuple[float, float]): The pitch min/max range.
            speakers (torch.Tensor): Tensor of speaker identities.
            langs (torch.Tensor): Tensor of language identities.
            p_control (float): Pitch control parameter. Defaults to 1.0.
            d_control (float): Duration control parameter. Defaults to 1.0.

        Returns:
            torch.Tensor: Predicted mel spectrogram.
        """
        # Generate masks for padding positions in the source sequences
        src_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64),
        ).to(x.device)

        # Obtain the embeddings for the input
        x, embeddings = self.get_embeddings(
            token_idx=x, speaker_idx=speakers, src_mask=src_mask, lang_idx=langs,
        )

        # Generate positional encodings
        encoding = positional_encoding(self.emb_dim, x.shape[1]).to(x.device)

        # Process the embeddings through the encoder
        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        # Predict prosody at utterance level and phoneme level
        u_prosody_pred = self.u_norm(
            self.average_utterance_prosody(
                u_prosody_pred=self.utterance_prosody_predictor(x=x, mask=src_mask),
                src_mask=src_mask,
            ),
        )
        p_prosody_pred = self.p_norm(
            self.phoneme_prosody_predictor(
                x=x,
                mask=src_mask,
            ),
        )
        x = x + self.u_bottle_out(u_prosody_pred).expand_as(x)
        x = x + self.p_bottle_out(p_prosody_pred).expand_as(x)
        x_res = x

        # NOTE: changed the pitch adaptor to the new one
        # x = self.pitch_adaptor.add_pitch(
        #     x=x, pitch_range=pitches_range, src_mask=src_mask, control=p_control,
        # )

        x, _ = self.pitch_adaptor_conv.add_pitch_embedding(
            x=x,
            mask=src_mask,
        )

        x, _ = self.energy_adaptor.add_energy_embedding(
            x=x,
            mask=src_mask,
        )

        x, _, embeddings = self.length_regulator.upsample(
            x=x,
            x_res=x_res,
            src_mask=src_mask,
            control=d_control,
            embeddings=embeddings,
        )

        mel_mask = tools.get_mask_from_lengths(
            torch.tensor([x.shape[1]], dtype=torch.int64),
        ).to(x.device)

        if x.shape[1] > encoding.shape[1]:
            encoding = positional_encoding(self.emb_dim, x.shape[1]).to(x.device)

        x = self.decoder(x, mel_mask, embeddings=embeddings, encoding=encoding)

        # x = self.to_mel_conv(x.permute((1, 2, 0)))
        x = self.to_mel(x.permute((1, 2, 0)))
        return x.permute((2, 1, 0))
