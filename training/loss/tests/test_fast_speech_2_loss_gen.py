import unittest

import torch

from models.config import PreprocessingConfig
from training.loss.fast_speech_2_loss_gen import FastSpeech2LossGen


class TestFastSpeech2LossGen(unittest.TestCase):
    def setUp(self):
        self.preprocessing_config = PreprocessingConfig("english_only")

        self.loss_gen = FastSpeech2LossGen()

    def test_forward(self):
        # Reproducible results
        torch.random.manual_seed(0)

        # Test with all inputs of shape (1, 11)
        src_masks = torch.zeros((1, 11), dtype=torch.bool)
        mel_masks = torch.zeros((1, 11), dtype=torch.bool)
        mel_targets = torch.randn((1, 11, 11))
        # postnet = torch.randn((1, 11, 11))
        mel_predictions = torch.randn((1, 11, 11))
        log_duration_predictions = torch.randn((1, 11))
        u_prosody_ref = torch.randn((1, 11))
        u_prosody_pred = torch.randn((1, 11))
        p_prosody_ref = torch.randn((1, 11, 11))
        p_prosody_pred = torch.randn((1, 11, 11))
        durations = torch.randn((1, 11))
        pitch_predictions = torch.randn((1, 11))
        p_targets = torch.randn((1, 11))
        attn_logprob = torch.randn((1, 1, 11, 11))
        attn_soft = torch.randn((1, 11, 11))
        attn_hard = torch.randn((1, 11, 11))
        step = 20000
        src_lens = torch.ones((1,), dtype=torch.long)
        mel_lens = torch.ones((1,), dtype=torch.long)
        energy_pred = torch.randn((1, 11))
        energy_target = torch.randn((1, 11))

        (
            total_loss,
            mel_loss,
            # mel_loss_postnet,
            ssim_loss,
            # ssim_loss_postnet,
            duration_loss,
            u_prosody_loss,
            p_prosody_loss,
            pitch_loss,
            ctc_loss,
            bin_loss,
            energy_loss,
        ) = self.loss_gen.forward(
            src_masks,
            mel_masks,
            mel_targets,
            mel_predictions,
            # postnet,
            log_duration_predictions,
            u_prosody_ref,
            u_prosody_pred,
            p_prosody_ref,
            p_prosody_pred,
            durations,
            pitch_predictions,
            p_targets,
            attn_logprob,
            attn_soft,
            attn_hard,
            step,
            src_lens,
            mel_lens,
            energy_pred,
            energy_target,
        )

        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(mel_loss, torch.Tensor)
        # self.assertIsInstance(mel_loss_postnet, torch.Tensor)
        self.assertIsInstance(ssim_loss, torch.Tensor)
        # self.assertIsInstance(ssim_loss_postnet, torch.Tensor)
        self.assertIsInstance(duration_loss, torch.Tensor)
        self.assertIsInstance(u_prosody_loss, torch.Tensor)
        self.assertIsInstance(p_prosody_loss, torch.Tensor)
        self.assertIsInstance(pitch_loss, torch.Tensor)
        self.assertIsInstance(ctc_loss, torch.Tensor)
        self.assertIsInstance(bin_loss, torch.Tensor)
        self.assertIsInstance(energy_loss, torch.Tensor)

        # Assert the value of losses
        self.assertTrue(
            torch.all(
                torch.tensor(
                    [
                        total_loss,
                        mel_loss,
                        # mel_loss_postnet,
                        ssim_loss,
                        # ssim_loss_postnet,
                        duration_loss,
                        u_prosody_loss,
                        p_prosody_loss,
                        pitch_loss,
                        ctc_loss,
                        bin_loss,
                        energy_loss,
                    ],
                ) >= 0,
            ),
        )


if __name__ == "__main__":
    unittest.main()
