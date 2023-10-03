import unittest
from unittest.mock import Mock

import torch

from model.acoustic_model.length_adaptor import LengthAdaptor
from model.helpers.tools import get_device


class TestLengthAdaptor(unittest.TestCase):
    def setUp(self):
        self.device = get_device()

        mock_model_config = Mock()

        # Attributes for model_config.encoder
        mock_model_config.encoder.n_hidden = 512

        # Attributes for model_config.variance_adaptor
        mock_model_config.variance_adaptor.n_hidden = 512
        mock_model_config.variance_adaptor.kernel_size = 5
        mock_model_config.variance_adaptor.p_dropout = 0.5

        self.model_config = mock_model_config

        self.length_adaptor = LengthAdaptor(self.model_config, device=self.device)
        self.batch_size = 2
        self.seq_length = 5
        self.n_hidden = 512  # should match cls.model_config.encoder.n_hidden

        self.x = torch.rand(
            self.batch_size, self.seq_length, self.n_hidden, device=self.device
        )
        self.x_res = torch.rand(
            self.batch_size, self.seq_length, self.n_hidden, device=self.device
        )
        self.src_mask = torch.ones(
            self.batch_size, self.seq_length, dtype=torch.bool, device=self.device
        )

    def test_length_regulate(self):
        duration = torch.full(
            (self.batch_size, self.seq_length), fill_value=2.0, device=self.device
        )
        output, mel_len = self.length_adaptor.length_regulate(self.x, duration)

        # Assert the device type
        self.assertEqual(output.device.type, self.device.type)
        self.assertEqual(mel_len.device.type, self.device.type)

        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.is_tensor(mel_len))

    def test_expand(self):
        predicted = torch.randint(
            low=0, high=2, size=(self.seq_length,), device=self.device
        )
        out = self.length_adaptor.expand(self.x[0], predicted)

        # Assert the device type
        self.assertEqual(out.device.type, self.device.type)

        self.assertTrue(torch.is_tensor(out))

        # Getting the sum of predicted tensor values, which will be the expected dimension size after expand
        # Assuming dimension 0 (time-steps) is being expanded
        expected_dim_0 = predicted.sum().item()
        actual_dim_0 = out.size(0)

        # Check if the size of expanded dimension matches with the sum of predicted values.
        self.assertEqual(expected_dim_0, actual_dim_0)

        # Check the size of unexpanded dimensions
        self.assertEqual(self.x.size(1), self.seq_length)
        self.assertEqual(out.size(1), self.n_hidden)

    def test_upsample_train(self):
        duration_target = torch.full(
            (self.batch_size, self.seq_length), fill_value=2.0, device=self.device
        )
        embeddings = torch.rand(
            self.batch_size, self.seq_length, self.n_hidden, device=self.device
        )

        x, log_duration_prediction, new_embeddings = self.length_adaptor.upsample_train(
            self.x, self.x_res, duration_target, embeddings, self.src_mask
        )

        # Assert the device type
        self.assertEqual(x.device.type, self.device.type)
        self.assertEqual(log_duration_prediction.device.type, self.device.type)
        self.assertEqual(new_embeddings.device.type, self.device.type)

        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_tensor(log_duration_prediction))
        self.assertTrue(torch.is_tensor(new_embeddings))

        # Check the size of tensor x, log_duration_prediction and new_embeddings
        expected_dim_0 = duration_target.sum().item()

        new_size_coef = self.batch_size * self.seq_length

        self.assertEqual(expected_dim_0, x.size(0) * new_size_coef)
        self.assertEqual(self.x.size(1), self.seq_length)
        self.assertEqual(x.size(2), self.n_hidden)

        self.assertEqual(log_duration_prediction.size(0), self.batch_size)
        self.assertEqual(log_duration_prediction.size(1), self.seq_length)

        self.assertEqual(expected_dim_0, new_embeddings.size(0) * new_size_coef)

        self.assertEqual(new_embeddings.size(1), self.seq_length * self.batch_size)
        self.assertEqual(new_embeddings.size(2), self.n_hidden)

    def test_upsample(self):
        embeddings = torch.rand(self.batch_size, self.seq_length, self.n_hidden)
        control = 0.5
        x, duration_rounded, new_embeddings = self.length_adaptor.upsample(
            self.x, self.x_res, self.src_mask, embeddings, control
        )

        # Assert the device type
        self.assertEqual(x.device.type, self.device.type)
        self.assertEqual(duration_rounded.device.type, self.device.type)
        self.assertEqual(new_embeddings.device.type, self.device.type)

        self.assertTrue(torch.is_tensor(x))
        self.assertTrue(torch.is_tensor(duration_rounded))
        self.assertTrue(torch.is_tensor(new_embeddings))

        # Check the size of tensor x, duration_rounded and new_embeddings
        self.assertEqual(x.size(0), self.batch_size)
        self.assertEqual(self.x.size(1), self.seq_length)
        self.assertEqual(x.size(2), self.n_hidden)

        self.assertEqual(duration_rounded.size(0), self.batch_size)
        self.assertEqual(duration_rounded.size(1), self.seq_length)

        self.assertEqual(new_embeddings.size(0), self.batch_size)
        # Always different?
        # self.assertEqual(new_embeddings.size(1), 1)
        self.assertEqual(new_embeddings.size(2), self.n_hidden)


if __name__ == "__main__":
    unittest.main()
