import unittest

import torch

from training.loss import UnivnetLoss


class TestUnivnetLoss(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(42)
        self.loss_module = UnivnetLoss()

    def test_forward(self):
        # Create some fake input data
        audio = torch.randn(1, 1, 22050)
        fake_audio = torch.randn(1, 1, 22050)
        res_fake = [(torch.randn(1, 1, 22050), torch.randn(1))]
        period_fake = [(torch.randn(1, 1, 22050), torch.randn(1))]
        res_real = [(torch.randn(1, 1, 22050), torch.randn(1))]
        period_real = [(torch.randn(1, 1, 22050), torch.randn(1))]

        # Call the forward method
        output = self.loss_module.forward(
            audio, fake_audio, res_fake, period_fake, res_real, period_real,
        )

        # Check that the output is a tuple with the expected lens
        self.assertIsInstance(output, tuple)

        self.assertEqual(len(output), 6)

        (
            total_loss_gen,
            total_loss_disc,
            stft_loss,
            score_loss,
            esr_loss,
            snr_loss,
        ) = output

        self.assertIsInstance(total_loss_gen, torch.Tensor)
        self.assertIsInstance(total_loss_disc, torch.Tensor)
        self.assertIsInstance(stft_loss, torch.Tensor)
        self.assertIsInstance(score_loss, torch.Tensor)
        self.assertIsInstance(esr_loss, torch.Tensor)
        self.assertIsInstance(snr_loss, torch.Tensor)

        # Assert the value of losses
        self.assertTrue(
            torch.all(
                torch.tensor(
                    [
                        total_loss_gen,
                        total_loss_disc,
                        stft_loss,
                        score_loss,
                        esr_loss,
                        snr_loss,
                    ],
                ) >= 0,
            ),
        )

if __name__ == "__main__":
    unittest.main()
