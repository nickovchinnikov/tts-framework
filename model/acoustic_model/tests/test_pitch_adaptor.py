import unittest
from unittest.mock import Mock

import torch

from model.acoustic_model import PitchAdaptor


class TestPitchAdaptor(unittest.TestCase):
    def setUp(self):
        # Mock object simulates the config object
        model_config = Mock()
        model_config.encoder.n_hidden = 512
        model_config.variance_adaptor.n_hidden = 256
        model_config.variance_adaptor.kernel_size = 3
        model_config.variance_adaptor.p_dropout = 0.3
        model_config.variance_adaptor.n_bins = 10
        self.pitch_adaptor = PitchAdaptor(
            model_config, "./model/acoustic_model/tests/mocks"
        )

        # Create a mock tensor for the inputs
        self.x = torch.rand(2, 10, 512)
        self.pitch_target = torch.rand(2, 10)
        self.src_mask_empty = torch.zeros(2, 10).type(torch.bool)
        self.src_mask_filled = torch.ones(2, 10).type(torch.bool)

    def test_add_pitch_train(self):
        # Execute `add_pitch_train()` method
        (
            out,
            pitch_prediction,
            pitch_embedding_true,
            pitch_embedding_pred,
        ) = self.pitch_adaptor.add_pitch_train(
            self.x, self.pitch_target, self.src_mask_empty, use_ground_truth=True
        )

        # Validate shapes
        self.assertEqual(out.shape, self.x.shape)
        self.assertEqual(pitch_prediction.shape, self.pitch_target.shape)
        self.assertEqual(pitch_embedding_true.shape, self.x.shape)
        self.assertEqual(pitch_embedding_pred.shape, self.x.shape)

    def test_add_pitch(self):
        # Execute `add_pitch()` method
        out = self.pitch_adaptor.add_pitch(self.x, self.src_mask_empty, control=1.0)

        # Validate the output shape
        self.assertEqual(out.shape, self.x.shape)

    def test_diff_input_output(self):
        # Inputs
        out = self.pitch_adaptor.add_pitch(self.x, self.src_mask_empty, control=1.0)

        # Check the output tensor is different from input tensor
        self.assertFalse(torch.equal(out, self.x))

    def test_effect_of_mask(self):
        # Create a mask tensor of all False (no position is masked)
        mask = torch.zeros_like(self.src_mask_empty).bool()

        # Call add_pitch() with mask of all False
        out = self.pitch_adaptor.add_pitch(self.x, mask, control=1.0)

        # There should be no masked positions in the output tensor
        # Assert those positions are equal to the corresponding input tensor's positions
        self.assertTrue(torch.all(out[mask] == self.x[mask]))


if __name__ == "__main__":
    unittest.main()
