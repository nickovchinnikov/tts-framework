import unittest

import torch

from model.helpers.initializer import (
    get_test_configs,
    init_acoustic_model,
    init_forward_trains_params,
)

# TODO: profile deeply the memory usage
# from torch.profiler import profile, record_function, ProfilerActivity
import model.helpers.tools as tools
from model.helpers.tools import get_device


# AcousticModel test
# Integration test
class TestAcousticModel(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        # TODO: optimize the model, so that it can be tested with srink_factor=1
        # Get config with srink_factor=4
        # Memory error with srink_factor =< 2
        # Probably because of the size of the model, probably memory leak
        # Tried to profile the memory usage, but it didn't help
        (
            self.preprocess_config,
            self.model_config,
            self.acoustic_pretraining_config,
        ) = get_test_configs(srink_factor=4)

        # Based on speaker.json mock
        n_speakers = 10

        # Add AcousticModel instance
        self.acoustic_model, _ = init_acoustic_model(
            self.preprocess_config, self.model_config, n_speakers, device=self.device
        )

        # Generate mock data for the forward pass
        self.forward_train_params = init_forward_trains_params(
            self.model_config,
            self.acoustic_pretraining_config,
            self.preprocess_config,
            n_speakers,
            device=self.device,
        )

    def test_get_embeddings(self):
        # Generate masks for padding positions in the source sequences and mel sequences
        # src_mask: Tensor containing the masks for padding positions in the source sequences. Shape: [1, batch_size]
        src_mask = tools.get_mask_from_lengths(self.forward_train_params.src_lens)

        # token_embeddings: Tensor containing the input sequences. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim]
        # embeddings: Tensor containing the embeddings. Shape: [speaker_embed_dim, batch_size, speaker_embed_dim + lang_embed_dim]
        token_embeddings, embeddings = self.acoustic_model.get_embeddings(
            token_idx=self.forward_train_params.x,
            speaker_idx=self.forward_train_params.speakers,
            src_mask=src_mask,
            lang_idx=self.forward_train_params.langs,
        )

        # Assert the device type
        self.assertEqual(token_embeddings.device.type, self.device.type)
        self.assertEqual(embeddings.device.type, self.device.type)

        self.assertEqual(
            token_embeddings.shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.acoustic_pretraining_config.batch_size,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )
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

    def test_forward_train(self):
        # with profile(
        #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #     with_stack=True,
        #     profile_memory=True,
        # ) as prof:
        result = self.acoustic_model.forward_train(
            x=self.forward_train_params.x,
            speakers=self.forward_train_params.speakers,
            src_lens=self.forward_train_params.src_lens,
            mels=self.forward_train_params.mels,
            mel_lens=self.forward_train_params.mel_lens,
            pitches=self.forward_train_params.pitches,
            langs=self.forward_train_params.langs,
            # TODO: add attn_priors check
            attn_priors=None,
            # TODO: add use_ground_truth check
            use_ground_truth=True,
        )
        # print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))

        # Assert the device type
        for key, value in result.items():
            self.assertEqual(
                value.device.type, self.device.type, f"Device type mismatch for {key}"
            )

        self.assertEqual(
            result["y_pred"].shape,
            torch.Size(
                [
                    self.model_config.speaker_embed_dim,
                    self.preprocess_config.stft.n_mel_channels,
                    self.model_config.speaker_embed_dim,
                ]
            ),
        )

        pitch_shape = torch.Size(
            [
                self.model_config.speaker_embed_dim,
                self.acoustic_pretraining_config.batch_size,
            ]
        )

        self.assertEqual(
            result["pitch_prediction"].shape,
            pitch_shape,
        )
        self.assertEqual(result["pitch_target"].shape, pitch_shape)
        self.assertEqual(result["log_duration_prediction"].shape, pitch_shape)

        prosody_shape = torch.Size(
            [
                self.model_config.speaker_embed_dim,
                self.model_config.lang_embed_dim,
                self.model_config.reference_encoder.bottleneck_size_u,
            ]
        )

        self.assertEqual(result["u_prosody_pred"].shape, prosody_shape)
        self.assertEqual(result["u_prosody_ref"].shape, prosody_shape)

        bottle_shape = torch.Size(
            [
                self.model_config.speaker_embed_dim,
                self.acoustic_pretraining_config.batch_size,
                self.model_config.reference_encoder.bottleneck_size_p,
            ]
        )

        self.assertEqual(result["p_prosody_pred"].shape, bottle_shape)
        self.assertEqual(result["p_prosody_ref"].shape, bottle_shape)

        attn_shape = torch.Size(
            [
                self.model_config.speaker_embed_dim,
                1,
                self.model_config.speaker_embed_dim,
                self.acoustic_pretraining_config.batch_size,
            ]
        )

        self.assertEqual(result["attn_logprob"].shape, attn_shape)
        self.assertEqual(result["attn_soft"].shape, attn_shape)
        self.assertEqual(result["attn_hard"].shape, attn_shape)

        self.assertEqual(result["attn_hard_dur"].shape, pitch_shape)

    def test_average_utterance_prosody(self):
        u_prosody_pred = torch.randn(
            2, 5, self.model_config.encoder.n_hidden, device=self.device
        )
        src_mask = torch.tensor(
            [[False, True, True, True, True], [False, False, True, True, True]],
            device=self.device,
        )

        averaged_prosody_pred = self.acoustic_model.average_utterance_prosody(
            u_prosody_pred=u_prosody_pred, src_mask=src_mask
        )

        self.assertEqual(
            averaged_prosody_pred.shape,
            torch.Size([2, 1, self.model_config.encoder.n_hidden]),
        )

    def test_forward(self):
        x = self.acoustic_model.forward(
            x=self.forward_train_params.x,
            speakers=self.forward_train_params.speakers,
            langs=self.forward_train_params.langs,
            p_control=0.5,
            d_control=0.5,
        )

        # Assert the device type
        self.assertEqual(x.device.type, self.device.type)

        # The last dim is not stable!
        self.assertEqual(x.shape[0], self.model_config.speaker_embed_dim)
        self.assertEqual(x.shape[1], self.preprocess_config.stft.n_mel_channels)


if __name__ == "__main__":
    unittest.main()
