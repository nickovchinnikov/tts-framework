import torch

from dataclasses import dataclass

from config import (
    AcousticModelConfigType,
    PreprocessingConfig,
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    SUPPORTED_LANGUAGES,
)

from model.acoustic_model import AcousticModel

from model.attention.conformer import Conformer


@dataclass
class ConformerConfig:
    dim: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    p_dropout: float
    kernel_size_conv_mod: int
    with_ff: bool


# Function to initialize a Conformer with a given AcousticModelConfigType configuration
def init_conformer(model_config: AcousticModelConfigType) -> Conformer:
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

    return Conformer(**vars(conformer_config))


@dataclass
class AcousticModelConfig:
    data_path: str
    preprocess_config: PreprocessingConfig
    model_config: AcousticENModelConfig
    fine_tuning: bool
    n_speakers: int


def init_acoustic_model(
    preprocess_config: PreprocessingConfig,
    model_config: AcousticENModelConfig,
    n_speakers: int = 10,
) -> AcousticModel:
    # Create an AcousticModelConfig instance
    acoustic_model_config = AcousticModelConfig(
        data_path="./model/acoustic_model/tests/mocks",
        preprocess_config=preprocess_config,
        model_config=model_config,
        fine_tuning=True,
        n_speakers=n_speakers,
    )

    return AcousticModel(**vars(acoustic_model_config))


@dataclass
class ForwardTrainParams:
    x: torch.Tensor
    speakers: torch.Tensor
    src_lens: torch.Tensor
    mels: torch.Tensor
    mel_lens: torch.Tensor
    pitches: torch.Tensor
    langs: torch.Tensor
    attn_priors: torch.Tensor
    use_ground_truth: bool = True


def init_forward_trains_params(
    model_config: AcousticENModelConfig,
    acoustic_pretraining_config: AcousticPretrainingConfig,
    n_speakers: int = 10,
) -> ForwardTrainParams:
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
        # speakers: Tensor containing the speaker indices. Shape: [speaker_embed_dim, batch_size]
        speakers=torch.randint(
            1,
            n_speakers - 1,
            (
                model_config.speaker_embed_dim,
                acoustic_pretraining_config.batch_size,
            ),
        ),
        # src_lens: Tensor containing the lengths of source sequences. Shape: [batch_size]
        src_lens=torch.tensor([acoustic_pretraining_config.batch_size]),
        # mels: Tensor containing the mel spectrogram. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
        mels=torch.randn(
            acoustic_pretraining_config.batch_size,
            model_config.speaker_embed_dim,
            model_config.encoder.n_hidden,
        ),
        # mel_lens: Tensor containing the lengths of mel sequences. Shape: [batch_size]
        mel_lens=torch.randint(
            0,
            model_config.speaker_embed_dim,
            (acoustic_pretraining_config.batch_size,),
        ),
        # pitches: Tensor containing the pitch values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
        pitches=torch.randn(
            acoustic_pretraining_config.batch_size,
            model_config.speaker_embed_dim,
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
            acoustic_pretraining_config.batch_size,
            model_config.speaker_embed_dim,
            model_config.speaker_embed_dim,
        ),
        use_ground_truth=True,
    )
