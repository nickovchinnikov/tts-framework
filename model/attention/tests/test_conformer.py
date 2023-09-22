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
    symbols,
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


# @inprogress
# @todo: it's one of the most important component test
# But it's too complicated to cover it from the first glance.
# You need to come back here when acoustic model is ready!
class TestConformer(unittest.TestCase):
    def setUp(self):
        self.acoustic_pretraining_config = AcousticPretrainingConfig()
        self.model_config = AcousticENModelConfig()
        self.preprocess_config = PreprocessingConfig("english_only")

        # Based on speaker.json mock
        n_speakers = 10

        self.conformer_config = ConformerConfig(
            dim=self.model_config.encoder.n_hidden,
            n_layers=self.model_config.encoder.n_layers,
            n_heads=self.model_config.encoder.n_heads,
            embedding_dim=self.model_config.speaker_embed_dim
            + self.model_config.lang_embed_dim,
            p_dropout=self.model_config.encoder.p_dropout,
            kernel_size_conv_mod=self.model_config.encoder.kernel_size_conv_mod,
            with_ff=self.model_config.encoder.with_ff,
        )

        # Convert ConformerConfig to a dictionary
        self.encoder = Conformer(**vars(self.conformer_config))

        self.acoustic_model_config = AcousticModelConfig(
            data_path="./model/acoustic_model/tests/mocks",
            preprocess_config=self.preprocess_config,
            model_config=self.model_config,
            fine_tuning=True,
            n_speakers=n_speakers,
        )

        self.acoustic_model = AcousticModel(**vars(self.acoustic_model_config))

        # Generate mock data for the forward pass
        self.forward_train_params = ForwardTrainParams(
            x=torch.randint(
                1,
                255,
                (
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    # self.acoustic_pretraining_config.batch_size,
                    # self.model_config.speaker_embed_dim,
                    # self.acoustic_pretraining_config.batch_size,
                ),
            ),
            # speakers=torch.tensor([[2, 0, 6], [9, 2, 3], [4, 5, 6]]),
            speakers=torch.randint(
                1,
                n_speakers - 1,
                (
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    # self.acoustic_pretraining_config.batch_size,
                    # self.model_config.speaker_embed_dim,
                    # self.acoustic_pretraining_config.batch_size,
                ),
            ),
            src_lens=torch.tensor([3]),
            mels=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.encoder.n_hidden,
            ),
            mel_lens=torch.randint(
                0,
                self.model_config.speaker_embed_dim,
                (self.acoustic_pretraining_config.batch_size,),
            ),
            pitches=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.encoder.n_hidden,
            ),
            # langs=torch.randint(
            #     1,
            #     len(SUPPORTED_LANGUAGES) - 1,
            #     (
            #         self.acoustic_pretraining_config.batch_size,
            #         self.model_config.speaker_embed_dim,
            #         # self.acoustic_pretraining_config.batch_size,
            #         # self.model_config.speaker_embed_dim,
            #         # self.acoustic_pretraining_config.batch_size,
            #     ),
            # ),
            langs=torch.randint(
                1,
                len(SUPPORTED_LANGUAGES) - 1,
                (
                    self.model_config.speaker_embed_dim,
                    # self.model_config.lang_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                ),
            ),
            # langs=torch.tensor([[3, 4, 5], [9, 2, 3], [4, 5, 6]]),
            attn_priors=torch.randn(
                self.acoustic_pretraining_config.batch_size,
                self.model_config.speaker_embed_dim,
                self.model_config.speaker_embed_dim,
            ),
            use_ground_truth=True,
        )
        self.src_word_emb = Parameter(
            tools.initialize_embeddings(
                (len(symbols), self.model_config.encoder.n_hidden)
            )
        )

        self.speaker_embed = Parameter(
            tools.initialize_embeddings(
                (n_speakers, self.model_config.speaker_embed_dim)
            )
        )
        self.lang_embed = Parameter(
            tools.initialize_embeddings(
                (len(SUPPORTED_LANGUAGES), self.model_config.lang_embed_dim)
            )
        )

    def test_initialization(self):
        """
        Test that a Conformer instance is correctly initialized.
        """
        self.assertIsInstance(self.model, Conformer)

    def test_check(self):
        src_mask = tools.get_mask_from_lengths(self.forward_train_params.src_lens)
        n_speakers = 10

        token_idx = torch.randint(
            1,
            255,
            (
                self.model_config.speaker_embed_dim,
                self.acoustic_pretraining_config.batch_size,
            ),
        )

        speaker_idx = torch.randint(
            1,
            n_speakers - 1,
            (
                self.model_config.speaker_embed_dim,
                self.acoustic_pretraining_config.batch_size,
            ),
        )

        lang_idx = torch.randint(
            1,
            len(SUPPORTED_LANGUAGES) - 1,
            (
                self.model_config.speaker_embed_dim,
                # self.model_config.lang_embed_dim,
                self.acoustic_pretraining_config.batch_size,
            ),
        )

        # speaker_idx = torch.tensor([[2, 0, 6], [9, 2, 3], [4, 5, 6]])
        # lang_idx = torch.tensor([[3, 4, 5], [9, 2, 3], [4, 5, 6]])

        # token_idx.shape == torch.Size([384, 3])
        # self.src_word_emb.shape == torch.Size([256, 384])
        # token_embeddings.shape == torch.Size([384, 3, 384])
        token_embeddings = F.embedding(token_idx, self.src_word_emb)

        # speaker_idx.shape == torch.Size([384, 3])
        # self.speaker_embed == torch.Size([10, 384])
        # speaker_embeds.shape == torch.Size([384, 3, 384])
        speaker_embeds = F.embedding(speaker_idx, self.speaker_embed)

        # #1 lang_idx.shape == torch.Size([3, 3])
        # self.lang_embed == torch.Size([19, 1])
        # lang_embeds.shape == torch.Size([3, 3, 1])

        # Check this out!
        # lang_idx.shape
        # > torch.Size([384, 3])
        # self.lang_embed == torch.Size([19, 1])
        # lang_embeds.shape
        # > torch.Size([384, 3, 1])
        #
        lang_embeds = F.embedding(lang_idx, self.lang_embed)

        # Merge the speaker and language embeddings
        embeddings = torch.cat([speaker_embeds, lang_embeds], dim=2)

        # Apply the mask to the embeddings and token embeddings
        embeddings = embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)
        token_embeddings = token_embeddings.masked_fill(src_mask.unsqueeze(-1), 0.0)

    def test_forward(self):
        """
        Test that a Conformer instance can correctly perform a forward pass.
        For this test case we use the code from AccousticModel.
        """
        # Generate masks for padding positions in the source sequences and mel sequences
        src_mask = tools.get_mask_from_lengths(self.forward_train_params.src_lens)

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

        # encoding.shape
        # > torch.Size([1, 287, 384])
        encoding = positional_encoding(
            self.model_config.encoder.n_hidden,
            max(x.shape[1], max(self.forward_train_params.mel_lens)),
            device=x.device,
        )

        # Run conformer encoder
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
