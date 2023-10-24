from unittest import TestCase

import torch

from training.loss import LossesCriterionAcoustic


class TestLossesCriterionAcoustic(TestCase):
    def setUp(self):
        self.criterion = LossesCriterionAcoustic()

    def test_forward(self):
        torch.random.manual_seed(0)

        # Define input tensors
        # Define masks with 0 and 1 values
        src_mask = torch.randint(low=0, high=2, size=(1, 11, 11), dtype=torch.bool)
        mel_mask = torch.randint(low=0, high=2, size=(1, 11, 11), dtype=torch.bool)

        src_lens = torch.ones((1,), dtype=torch.long)
        mel_lens = torch.ones((1,), dtype=torch.long)

        mels = torch.randn((1, 11, 11))
        y_pred = torch.randn((1, 11, 11))

        log_duration_prediction = torch.randn((1, 11, 11))
        p_prosody_ref = torch.randn((1, 11, 11))
        p_prosody_pred = torch.randn((1, 11, 11))
        pitch_prediction = torch.randn((1, 11))

        outputs = {
            "u_prosody_ref": torch.randn((1, 11), dtype=torch.float32),
            "u_prosody_pred": torch.randn((1, 11), dtype=torch.float32),
            "pitch_target": torch.randn((1, 11), dtype=torch.float32),
            "attn_hard_dur": torch.abs(torch.randn((1, 11, 11), dtype=torch.float32)),
            "attn_logprob": torch.randn((1, 1, 11, 11), dtype=torch.float32),
            "attn_soft": torch.randn((1, 11, 11), dtype=torch.float32),
            "attn_hard": torch.randn((1, 11, 11), dtype=torch.float32),
        }
        step = 0

        # Call the forward method
        total_loss = self.criterion.forward(
            src_mask=src_mask,
            src_lens=src_lens,
            mel_mask=mel_mask,
            mel_lens=mel_lens,
            mels=mels,
            y_pred=y_pred,
            log_duration_prediction=log_duration_prediction,
            p_prosody_ref=p_prosody_ref,
            p_prosody_pred=p_prosody_pred,
            pitch_prediction=pitch_prediction,
            outputs=outputs,
            step=step,
        )

        # Assert the value of losses
        self.assertTrue(
            torch.allclose(
                torch.tensor(
                    [
                        total_loss,
                        self.criterion.reconstruction_loss.item(),
                        self.criterion.mel_loss.item(),
                        self.criterion.ssim_loss.item(),
                        self.criterion.duration_loss.item(),
                        self.criterion.u_prosody_loss.item(),
                        self.criterion.p_prosody_loss.item(),
                        self.criterion.pitch_loss.item(),
                        self.criterion.ctc_loss.item(),
                        self.criterion.bin_loss.item(),
                    ],
                ),
                torch.tensor(
                    [
                        6.2330,
                        6.2330,
                        1.2304,
                        0.8610,
                        1.1307,
                        0.4871,
                        0.5688,
                        1.7354,
                        0.2195,
                        0.0,
                    ],
                ),
                atol=1e-4,
            ),
        )
