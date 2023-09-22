import unittest

from dataclasses import dataclass

import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from config import (
    AcousticENModelConfig,
    AcousticPretrainingConfig,
    PreprocessingConfig,
    SUPPORTED_LANGUAGES,
)

from model.acoustic_model import AcousticModel

from model.attention.conformer import Conformer

import helpers.tools as tools

from model.acoustic_model.helpers import positional_encoding


@dataclass
class ConformerConfig:
    dim: int
    n_layers: int
    n_heads: int
    embedding_dim: int
    p_dropout: float
    kernel_size_conv_mod: int
    with_ff: bool


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


@dataclass
class AcousticModelConfig:
    data_path: str
    preprocess_config: PreprocessingConfig
    model_config: AcousticENModelConfig
    fine_tuning: bool
    n_speakers: int


# It's one of the most important component test
# Conformer is used in the encoder of the AccousticModel, crucial for the training
# Here you can understand the input and output shapes of the Conformer
class TestConformer(unittest.TestCase):
    def setUp(self):
        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        # Based on speaker.json mock
        n_speakers = 10

        # Create a ConformerConfig instance
        self.conformer_config = ConformerConfig(
            dim=self.model_config.encoder.n_hidden,
            n_layers=self.model_config.encoder.n_layers,
            n_heads=self.model_config.encoder.n_heads,
            embedding_dim=self.model_config.speaker_embed_dim
            + self.model_config.lang_embed_dim,  # speaker_embed_dim + lang_embed_dim = 385
            p_dropout=self.model_config.encoder.p_dropout,
            kernel_size_conv_mod=self.model_config.encoder.kernel_size_conv_mod,
            with_ff=self.model_config.encoder.with_ff,
        )

        # Add Conformer as encoder
        self.encoder = Conformer(**vars(self.conformer_config))

        # Create an AcousticModelConfig instance
        self.acoustic_model_config = AcousticModelConfig(
            data_path="./model/acoustic_model/tests/mocks",
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            fine_tuning=True,
            n_speakers=n_speakers,
        )

        # Add AcousticModel instance
        self.acoustic_model = AcousticModel(**vars(self.acoustic_model_config))

        # Generate mock data for the forward pass
        self.forward_train_params = ForwardTrainParams(
            # x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size]
            x=torch.randint(
                1,
                255,
                (
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ),
            ),
            # speakers: Tensor containing the speaker indices. Shape: [speaker_embed_dim, batch_size]
            speakers=torch.randint(
                1,
                n_speakers - 1,
                (
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ),
            ),
            # src_lens: Tensor containing the lengths of source sequences. Shape: [batch_size]
            src_lens=torch.tensor([self.acoustic_pretraining_config.batch_size]),
            # mels: Tensor containing the mel spectrogram. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
            mels=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.encoder.n_hidden,
            ),
            # mel_lens: Tensor containing the lengths of mel sequences. Shape: [batch_size]
            mel_lens=torch.randint(
                0,
                self.model_config.speaker_embed_dim,
                (self.acoustic_pretraining_config.batch_size,),
            ),
            # pitches: Tensor containing the pitch values. Shape: [batch_size, speaker_embed_dim, encoder.n_hidden]
            pitches=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.encoder.n_hidden,
            ),
            # langs: Tensor containing the language indices. Shape: [speaker_embed_dim, batch_size]
            langs=torch.randint(
                1,
                len(SUPPORTED_LANGUAGES) - 1,
                (
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ),
            ),
            # attn_priors: Tensor containing the attention priors. Shape: [batch_size, speaker_embed_dim, speaker_embed_dim]
            attn_priors=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.speaker_embed_dim,
            ),
            use_ground_truth=True,
        )

    def test_initialization(self):
        """
        Test that a Conformer instance is correctly initialized.
        """
        self.assertIsInstance(self.model, Conformer)

    def test_forward(self):
        """
        Test that a Conformer instance can correctly perform a forward pass.
        For this test case we use the code from AccousticModel.
        """
        # Generate masks for padding positions in the source sequences and mel sequences
        # src_mask: Tensor containing the masks for padding positions in the source sequences. Shape: [1, batch_size]
        src_mask = tools.get_mask_from_lengths(self.forward_train_params.src_lens)

        # x: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        # embeddings: Tensor containing the embeddings. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim + lang_embed_dim]
        x, embeddings = self.acoustic_model.get_embeddings(
            token_idx=self.forward_train_params.x,
            speaker_idx=self.forward_train_params.speakers,
            src_mask=src_mask,
            lang_idx=self.forward_train_params.langs,
        )

        # Assert the shape of x
        self.assertEqual(
            x.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )

        # Assert the shape of embeddings
        self.assertEqual(
            embeddings.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim
                    + self.model_config.lang_embed_dim,
                ]
            ),
        )

        # encoding: Tensor containing the positional encoding.
        # Shape: [lang_embed_dim, max(forward_train_params.mel_lens), encoder.n_hidden]
        encoding = positional_encoding(
            self.model_config.encoder.n_hidden,
            max(x.shape[1], max(self.forward_train_params.mel_lens)),
            device=x.device,
        )

        # Run conformer encoder
        # x: Tensor containing the encoded sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        x = self.encoder(x, src_mask, embeddings=embeddings, encoding=encoding)

        # Assert the shape of x
        self.assertEqual(
            x.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
