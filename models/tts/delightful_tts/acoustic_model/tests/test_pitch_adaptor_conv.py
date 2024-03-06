import unittest

import torch

from models.tts.delightful_tts.acoustic_model.pitch_adaptor_conv import PitchAdaptorConv


class TestPitchAdaptorConv(unittest.TestCase):
    def setUp(self):
        # Initialize common parameters for testing
        self.batch_size = 1
        self.seq_length = 11
        self.target_length = 58
        self.channels_in = 58
        self.channels_hidden = 58
        self.channels_out = 1
        self.kernel_size = 5
        self.dropout = 0.1
        self.leaky_relu_slope = 0.2
        self.emb_kernel_size = 3

        # Initialize the PitchAdaptorConv module
        self.pitch_adaptor = PitchAdaptorConv(
            channels_in=self.channels_in,
            channels_hidden=self.channels_hidden,
            channels_out=self.channels_out,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            leaky_relu_slope=self.leaky_relu_slope,
            emb_kernel_size=self.emb_kernel_size,
        )

        # Create sample input tensors
        self.inputs = torch.randn(self.batch_size, self.seq_length, self.channels_in)
        self.target = torch.randn(self.batch_size, 1, self.target_length)
        self.dr = torch.tensor([[ 5.,  5.,  5.,  5.,  4.,  5.,  5.,  4.,  5.,  5., 10.]])
        self.mask = torch.randint(1, self.seq_length, (self.batch_size, self.seq_length)).bool()

    def test_get_pitch_embedding_train(self):
        # Test get_pitch_embedding_train method
        pitch_pred, avg_pitch_target, pitch_emb = self.pitch_adaptor.get_pitch_embedding_train(
            x=self.inputs,
            target=self.target,
            dr=self.dr,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(pitch_pred.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(avg_pitch_target.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(pitch_emb.shape, (self.batch_size, self.channels_hidden, self.seq_length))

    def test_add_pitch_embedding_train(self):
        inputs = torch.randn(self.batch_size, self.seq_length, self.channels_in)
        target = torch.randn(self.batch_size, self.target_length)
        dr = torch.tensor([[ 5.,  5.,  5.,  5.,  4.,  5.,  5.,  4.,  5.,  5., 10.]])
        mask = torch.randint(1, self.seq_length, (self.batch_size, self.seq_length)).bool()

        # Test add_pitch_embedding_train method
        (
            x_with_pitch,
            pitch_pred,
            avg_pitch_target,
        ) = self.pitch_adaptor.add_pitch_embedding_train(
            x=inputs,
            target=target,
            dr=dr,
            mask=mask,
        )

        # Check shapes of output tensors
        self.assertEqual(x_with_pitch.shape, self.inputs.shape)
        self.assertEqual(pitch_pred.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(avg_pitch_target.shape, (self.batch_size, 1, self.seq_length))

    def test_get_pitch_embedding(self):
        # Test get_pitch_embedding method
        pitch_emb_pred, pitch_pred = self.pitch_adaptor.get_pitch_embedding(
            x=self.inputs,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(pitch_emb_pred.shape, (self.batch_size, self.channels_hidden, self.seq_length))
        self.assertEqual(pitch_pred.shape, (self.batch_size, 1, self.seq_length))

    def test_add_pitch_embedding(self):
        inputs = torch.randn(self.batch_size, self.seq_length, self.channels_in)
        mask = torch.randint(1, self.seq_length, (self.batch_size, self.seq_length)).bool()

        # Test add_pitch_embedding method
        x_with_pitch, pitch_pred = self.pitch_adaptor.add_pitch_embedding(
            x=inputs,
            mask=mask,
        )

        # Check shapes of output tensors
        self.assertEqual(x_with_pitch.shape, self.inputs.shape)
        self.assertEqual(pitch_pred.shape, (self.batch_size, 1, self.seq_length))

if __name__ == "__main__":
    unittest.main()
