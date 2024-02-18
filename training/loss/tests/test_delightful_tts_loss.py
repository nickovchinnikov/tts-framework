import unittest

import torch

from models.config import AcousticENModelConfig
from training.loss.delightful_tts_loss import (
    DelightfulTTSLoss,
    ForwardSumLoss,
    sample_wise_min_max,
    sequence_mask,
)


class TestLosses(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_sequence_mask(self):
        sequence_length = torch.tensor([2, 3, 4], device=self.device)
        max_len = 5
        mask = sequence_mask(sequence_length, max_len)
        expected_mask = torch.tensor([[True, True, False, False, False],
                                      [True, True, True, False, False],
                                      [True, True, True, True, False]], device=self.device)
        self.assertTrue(torch.equal(mask, expected_mask))

    def test_sample_wise_min_max(self):
        x = torch.tensor([[[1, 1], [1, 0], [0, 1]]], dtype=torch.float32, device=self.device)
        normalized_x = sample_wise_min_max(x)
        expected_normalized_x = torch.tensor([[
            [1., 1.],
            [1., 0.],
            [0., 1.],
        ]], device=self.device)
        self.assertTrue(torch.allclose(normalized_x, expected_normalized_x))

    def test_ForwardSumLoss(self):
        loss_function = ForwardSumLoss()
        attn_logprob = torch.randn((1, 1, 11, 11))
        src_lens = torch.ones((1,), dtype=torch.long)
        mel_lens = torch.ones((1,), dtype=torch.long)
        loss = loss_function(attn_logprob, src_lens, mel_lens)
        self.assertTrue(isinstance(loss, torch.Tensor))

    def test_DelightfulTTSLoss(self):
        model_config = AcousticENModelConfig()
        loss_function = DelightfulTTSLoss(model_config)
        mel_output = torch.randn((1, 11, 11))
        mel_target = torch.randn((1, 11, 11))
        mel_lens = torch.ones((1,), dtype=torch.long)
        dur_output = torch.randn((1, 11))
        dur_target = torch.randn((1, 11))
        pitch_output = torch.randn((1, 11))
        pitch_target = torch.randn((1, 11))
        energy_output = torch.randn((1, 11))
        energy_target = torch.randn((1, 11))
        src_lens = torch.ones((1,), dtype=torch.long)
        p_prosody_ref = torch.randn((1, 11, 11))
        p_prosody_pred = torch.randn((1, 11, 11))
        u_prosody_ref = torch.randn((1, 11, 11))
        u_prosody_pred = torch.randn((1, 11, 11))
        aligner_logprob = torch.randn((1, 1, 11, 11))
        aligner_hard = torch.randn((1, 11, 11))
        aligner_soft = torch.randn((1, 11, 11))
        total_loss, _, _, _, _, _, _, _, _, _ = loss_function(
            mel_output, mel_target, mel_lens, dur_output, dur_target, pitch_output, pitch_target,
            energy_output, energy_target, src_lens, p_prosody_ref, p_prosody_pred,
            u_prosody_ref, u_prosody_pred, aligner_logprob, aligner_hard, aligner_soft,
        )
        self.assertTrue(isinstance(total_loss, torch.Tensor))

if __name__ == "__main__":
    unittest.main()
