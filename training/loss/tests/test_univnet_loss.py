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
        output = self.loss_module(audio, fake_audio, res_fake, period_fake, res_real, period_real)

        # Check that the output is a tuple with the expected lens
        self.assertIsInstance(output, tuple)

        self.assertEqual(len(output), 4)

        (
            total_loss_gen,
            total_loss_disc,
            mel_loss,
            score_loss,
        ) = output

        self.assertAlmostEqual(total_loss_gen, 4.5500, places=4)
        self.assertAlmostEqual(total_loss_disc, 1.79197, places=4)
        self.assertAlmostEqual(mel_loss, 3.3992, places=4)
        self.assertAlmostEqual(score_loss, 1.1508, places=4)

if __name__ == "__main__":
    unittest.main()
