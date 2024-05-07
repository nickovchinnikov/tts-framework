from dataclasses import dataclass
from typing import Tuple

import torch

from models.config import (
    SUPPORTED_LANGUAGES,
    AcousticENModelConfig,
    AcousticModelConfigType,
    AcousticPretrainingConfig,
)
from models.config import (
    PreprocessingConfigUnivNet as PreprocessingConfig,
)
from models.helpers import positional_encoding, tools
from models.tts.delightful_tts.acoustic_model import AcousticModel
from models.tts.delightful_tts.attention.conformer import Conformer


@dataclass
class ConformerConfig:
    dim: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    p_dropout: float
    kernel_size_conv_mod: int
    with_ff: bool


def get_test_configs(
    srink_factor: int = 4,
) -> Tuple[PreprocessingConfig, AcousticENModelConfig, AcousticPretrainingConfig]:
    r"""Returns a tuple of configuration objects for testing purposes.

    Args:
        srink_factor (int, optional): The shrink factor to apply to the model configuration. Defaults to 4.

    Returns:
        Tuple[PreprocessingConfig, AcousticENModelConfig, AcousticPretrainingConfig]: A tuple of configuration objects for testing purposes.

    This function returns a tuple of configuration objects for testing purposes. The configuration objects are as follows:
    - `PreprocessingConfig`: A configuration object for preprocessing.
    - `AcousticENModelConfig`: A configuration object for the acoustic model.
    - `AcousticPretrainingConfig`: A configuration object for acoustic pretraining.

    The `srink_factor` parameter is used to shrink the dimensions of the model configuration to prevent out of memory issues during testing.
    """
    preprocess_config = PreprocessingConfig("english_only")
    model_config = AcousticENModelConfig()

    model_config.speaker_embed_dim = model_config.speaker_embed_dim // srink_factor
    model_config.encoder.n_hidden = model_config.encoder.n_hidden // srink_factor
    model_config.decoder.n_hidden = model_config.decoder.n_hidden // srink_factor
    model_config.variance_adaptor.n_hidden = (
        model_config.variance_adaptor.n_hidden // srink_factor
    )

    acoustic_pretraining_config = AcousticPretrainingConfig()

    return (preprocess_config, model_config, acoustic_pretraining_config)


# Function to initialize a Conformer with a given AcousticModelConfigType configuration
def init_conformer(
    model_config: AcousticModelConfigType,
) -> Tuple[Conformer, ConformerConfig]:
    r"""Function to initialize a `Conformer` with a given `AcousticModelConfigType` configuration.

    Args:
        model_config (AcousticModelConfigType): The object that holds the configuration details.

    Returns:
        Conformer: Initialized Conformer object.

    The function sets the details of the `Conformer` object based on the `model_config` parameter.
    The `Conformer` configuration is set as follows:
    - dim: The number of hidden units, taken from the encoder part of the `model_config.encoder.n_hidden`.
    - n_layers: The number of layers, taken from the encoder part of the `model_config.encoder.n_layers`.
    - n_heads: The number of attention heads, taken from the encoder part of the `model_config.encoder.n_heads`.
    - embedding_dim: The sum of dimensions of speaker embeddings and language embeddings.
      The speaker_embed_dim and lang_embed_dim are a part of the `model_config.speaker_embed_dim`.
    - p_dropout: Dropout rate taken from the encoder part of the `model_config.encoder.p_dropout`.
      It adds a regularization parameter to prevent overfitting.
    - kernel_size_conv_mod: The kernel size for the convolution module taken from the encoder part of the `model_config.encoder.kernel_size_conv_mod`.
    - with_ff: A Boolean value denoting if feedforward operation is involved, taken from the encoder part of the `model_config.encoder.with_ff`.

    """
    conformer_config = ConformerConfig(
        dim=model_config.encoder.n_hidden,
        n_layers=model_config.encoder.n_layers,
        n_heads=model_config.encoder.n_heads,
        embedding_dim=model_config.speaker_embed_dim
        + model_config.lang_embed_dim,  # speaker_embed_dim + lang_embed_dim = 385
        p_dropout=model_config.encoder.p_dropout,
        kernel_size_conv_mod=model_config.encoder.kernel_size_conv_mod,
        with_ff=model_config.encoder.with_ff,
    )

    model = Conformer(**vars(conformer_config))

    return model, conformer_config


@dataclass
class AcousticModelConfig:
    preprocess_config: PreprocessingConfig
    model_config: AcousticENModelConfig
    n_speakers: int


def init_acoustic_model(
    preprocess_config: PreprocessingConfig,
    model_config: AcousticENModelConfig,
    n_speakers: int = 10,
) -> Tuple[AcousticModel, AcousticModelConfig]:
    r"""Function to initialize an `AcousticModel` with given preprocessing and model configurations.

    Args:
        preprocess_config (PreprocessingConfig): Configuration object for pre-processing.
        model_config (AcousticENModelConfig): Configuration object for English Acoustic model.
        n_speakers (int, optional): Number of speakers. Defaults to 10.

    Returns:
        AcousticModel: Initialized Acoustic Model.

    The function creates an `AcousticModelConfig` instance which is then used to initialize the `AcousticModel`.
    The `AcousticModelConfig` is configured as follows:
    - preprocess_config: Pre-processing configuration.
    - model_config: English Acoustic model configuration.
    - fine_tuning: Boolean flag set to True indicating the model is for fine-tuning.
    - n_speakers: Number of speakers.

    """
    # Create an AcousticModelConfig instance
    acoustic_model_config = AcousticModelConfig(
        preprocess_config=preprocess_config,
        model_config=model_config,
        n_speakers=n_speakers,
    )

    model = AcousticModel(**vars(acoustic_model_config))

    return model, acoustic_model_config


@dataclass
class ForwardTrainParams:
    x: torch.Tensor
    speakers: torch.Tensor
    src_lens: torch.Tensor
    mels: torch.Tensor
    mel_lens: torch.Tensor
    enc_len: torch.Tensor
    pitches: torch.Tensor
    pitches_range: Tuple[float, float]
    energies: torch.Tensor
    langs: torch.Tensor
    attn_priors: torch.Tensor
    use_ground_truth: bool = True


def init_forward_trains_params(
    model_config: AcousticENModelConfig,
    acoustic_pretraining_config: AcousticPretrainingConfig,
    preprocess_config: PreprocessingConfig,
    n_speakers: int = 10,
) -> ForwardTrainParams:
    r"""Function to initialize the parameters for forward propagation during training.

    Args:
        model_config (AcousticENModelConfig): Configuration object for English Acoustic model.
        acoustic_pretraining_config (AcousticPretrainingConfig): Configuration object for acoustic pretraining.
        preprocess_config (PreprocessingConfig): Configuration object for pre-processing.
        n_speakers (int, optional): Number of speakers. Defaults to 10.

    Returns:
        ForwardTrainParams: Initialized parameters for forward propagation during training.

    The function initializes the ForwardTrainParams object with the following parameters:
    - x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size]
    - speakers: Tensor containing the speaker indices. Shape: [speaker_embed_dim, batch_size]
    - src_lens: Tensor containing the lengths of source sequences. Shape: [batch_size]
    - mels: Tensor containing the mel spectrogram. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
    - mel_lens: Tensor containing the lengths of mel sequences. Shape: [batch_size]
    - pitches: Tensor containing the pitch values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
    - energies: Tensor containing the energy values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
    - langs: Tensor containing the language indices. Shape: [speaker_embed_dim, batch_size]
    - attn_priors: Tensor containing the attention priors. Shape: [batch_size, speaker_embed_dim, speaker_embed_dim]
    - use_ground_truth: Boolean flag indicating if ground truth values should be used or not.

    All the Tensors are initialized with random values.
    """
    return ForwardTrainParams(
        # x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size]
        x=torch.randint(
            1,
            255,
            (
                model_config.speaker_embed_dim,
                acoustic_pretraining_config.batch_size,
            ),
        ),
        pitches_range=(0.0, 1.0),
        # speakers: Tensor containing the speaker indices. Shape: [speaker_embed_dim, batch_size]
        speakers=torch.randint(
            1,
            n_speakers - 1,
            (
                model_config.speaker_embed_dim,
                acoustic_pretraining_config.batch_size,
            ),
        ),
        # src_lens: Tensor containing the lengths of source sequences. Shape: [speaker_embed_dim]
        src_lens=torch.cat(
            [
                # torch.tensor([self.model_config.speaker_embed_dim]),
                torch.randint(
                    1,
                    acoustic_pretraining_config.batch_size + 1,
                    (model_config.speaker_embed_dim,),
                ),
            ],
            dim=0,
        ),
        # mels: Tensor containing the mel spectrogram. Shape: [batch_size, stft.n_mel_channels, encoder.n_hidden]
        mels=torch.randn(
            model_config.speaker_embed_dim,
            preprocess_config.stft.n_mel_channels,
            model_config.encoder.n_hidden,
        ),
        # enc_len: Tensor containing the lengths of mel sequences. Shape: [speaker_embed_dim]
        enc_len=torch.cat(
            [
                torch.randint(
                    1,
                    model_config.speaker_embed_dim,
                    (model_config.speaker_embed_dim - 1,),
                ),
                torch.tensor([model_config.speaker_embed_dim]),
            ],
            dim=0,
        ),
        # mel_lens: Tensor containing the lengths of mel sequences. Shape: [batch_size]
        mel_lens=torch.cat(
            [
                torch.randint(
                    1,
                    model_config.speaker_embed_dim,
                    (model_config.speaker_embed_dim - 1,),
                ),
                torch.tensor([model_config.speaker_embed_dim]),
            ],
            dim=0,
        ),
        # pitches: Tensor containing the pitch values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
        pitches=torch.randn(
            # acoustic_pretraining_config.batch_size,
            model_config.speaker_embed_dim,
            # model_config.speaker_embed_dim,
            model_config.encoder.n_hidden,
        ),
        # energies: Tensor containing the energy values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
        energies=torch.randn(
            model_config.speaker_embed_dim,
            1,
            model_config.encoder.n_hidden,
        ),
        # langs: Tensor containing the language indices. Shape: [speaker_embed_dim, batch_size]
        langs=torch.randint(
            1,
            len(SUPPORTED_LANGUAGES) - 1,
            (
                model_config.speaker_embed_dim,
                acoustic_pretraining_config.batch_size,
            ),
        ),
        # attn_priors: Tensor containing the attention priors. Shape: [batch_size, speaker_embed_dim, speaker_embed_dim]
        attn_priors=torch.randn(
            model_config.speaker_embed_dim,
            model_config.speaker_embed_dim,
            acoustic_pretraining_config.batch_size,
        ),
        use_ground_truth=True,
    )


def init_mask_input_embeddings_encoding_attn_mask(
    acoustic_model: AcousticModel,
    forward_train_params: ForwardTrainParams,
    model_config: AcousticENModelConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Function to initialize masks for padding positions, input sequences, embeddings, positional encoding and attention masks.

    Args:
        acoustic_model (AcousticModel): Initialized Acoustic Model.
        forward_train_params (ForwardTrainParams): Parameters for the forward training process.
        model_config (AcousticENModelConfig): Configuration object for English Acoustic model.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the following elements:
            - src_mask: Tensor containing the masks for padding positions in the source sequences. Shape: [1, batch_size]
            - x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
            - embeddings: Tensor containing the embeddings. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim + lang_embed_dim]
            - encoding: Tensor containing the positional encoding. Shape: [lang_embed_dim, max(forward_train_params.mel_lens), model_config.encoder.n_hidden]
            - attn_maskЖ Tensor containing the attention masks. Shape: [1, 1, 1, batch_size]

    The function starts by generating masks for padding positions in the source and mel sequences.
    Then, it uses the acoustic model to get the input sequences and embeddings.
    Finally, it computes the positional encoding.

    """
    # Generate masks for padding positions in the source sequences and mel sequences
    # src_mask: Tensor containing the masks for padding positions in the source sequences. Shape: [1, batch_size]
    src_mask = tools.get_mask_from_lengths(forward_train_params.src_lens)

    # x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
    # embeddings: Tensor containing the embeddings. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim + lang_embed_dim]
    x, embeddings = acoustic_model.get_embeddings(
        token_idx=forward_train_params.x,
        speaker_idx=forward_train_params.speakers,
        src_mask=src_mask,
        lang_idx=forward_train_params.langs,
    )

    # encoding: Tensor containing the positional encoding
    # Shape: [lang_embed_dim, max(forward_train_params.mel_lens), encoder.n_hidden]
    encoding = positional_encoding(
        model_config.encoder.n_hidden,
        max(x.shape[1], int(forward_train_params.mel_lens.max().item())),
    )

    attn_mask = src_mask.view((src_mask.shape[0], 1, 1, src_mask.shape[1]))

    return src_mask, x, embeddings, encoding, attn_mask
