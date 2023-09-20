import unittest
import torch

from unittest.mock import Mock 
from model.acoustic_model.length_adaptor import LengthAdaptor


class TestLengthAdaptor(unittest.TestCase):
    @classmethod
    def setUp(cls):
        mock_model_config = Mock()

        # Attributes for model_config.encoder
        mock_model_config.encoder.n_hidden = 512

        # Attributes for model_config.variance_adaptor
        mock_model_config.variance_adaptor.n_hidden = 512
        mock_model_config.variance_adaptor.kernel_size = 5
        mock_model_config.variance_adaptor.p_dropout = 0.5

        cls.model_config = mock_model_config

        cls.length_adaptor = LengthAdaptor(cls.model_config)
        cls.batch_size = 2
        cls.seq_length = 5
        cls.n_hidden = 512    # should match cls.model_config.encoder.n_hidden

        cls.x = torch.rand(cls.batch_size, cls.seq_length, cls.n_hidden)
        cls.x_res = torch.rand(cls.batch_size, cls.seq_length, cls.n_hidden)
        cls.src_mask = torch.ones(cls.batch_size, cls.seq_length)

    def test_length_regulate(self):
        duration = torch.full((self.batch_size, self.seq_length), fill_value=2.0)
        output, mel_len = self.length_adaptor.length_regulate(self.x, duration)        
        self.assertTrue(torch.is_tensor(output))
        self.assertTrue(torch.is_tensor(mel_len))

    def test_expand(self):
        predicted = torch.randint(low=0, high=2, size=(self.seq_length,))
        out = self.length_adaptor.expand(self.x[0], predicted)
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
        duration_target = torch.full((self.batch_size, self.seq_length), fill_value=2.0)
        embeddings = torch.rand(self.batch_size, self.seq_length, self.n_hidden)

        x, log_duration_prediction, new_embeddings = self.length_adaptor.upsample_train(
            self.x, self.x_res, duration_target, embeddings, self.src_mask
        )

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


if __name__ == '__main__':
    unittest.main()
