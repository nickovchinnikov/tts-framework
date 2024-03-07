import unittest

import torch

from models.tts.delightful_tts.acoustic_model.energy_adaptor import EnergyAdaptor


class TestEnergyAdaptor(unittest.TestCase):
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

        # Initialize the EnergyAdaptor module
        self.energy_adaptor = EnergyAdaptor(
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

    def test_get_energy_embedding_train(self):
        # Test get_energy_embedding_train method
        energy_pred, avg_energy_target, energy_emb = self.energy_adaptor.get_energy_embedding_train(
            x=self.inputs,
            target=self.target,
            dr=self.dr,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(energy_pred.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(avg_energy_target.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(energy_emb.shape, (self.batch_size, self.channels_hidden, self.seq_length))

    def test_add_energy_embedding_train(self):
        # Test add_energy_embedding_train method
        (
            x_with_energy,
            energy_pred,
            avg_energy_target,
        ) = self.energy_adaptor.add_energy_embedding_train(
            x=self.inputs,
            target=self.target,
            dr=self.dr,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(x_with_energy.shape, self.inputs.shape)
        self.assertEqual(energy_pred.shape, (self.batch_size, 1, self.seq_length))
        self.assertEqual(avg_energy_target.shape, (self.batch_size, 1, self.seq_length))

    def test_get_energy_embedding(self):
        # Initialize the EnergyAdaptor module
        energy_adaptor = EnergyAdaptor(
            channels_in=self.channels_in,
            channels_hidden=self.channels_hidden,
            channels_out=self.channels_out,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            leaky_relu_slope=self.leaky_relu_slope,
            emb_kernel_size=self.emb_kernel_size,
        )

        # Test get_energy_embedding method
        energy_emb_pred, energy_pred = energy_adaptor.get_energy_embedding(
            x=self.inputs,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(energy_emb_pred.shape, (self.batch_size, self.channels_hidden, self.seq_length))
        self.assertEqual(energy_pred.shape, (self.batch_size, 1, self.seq_length))

    def test_add_energy_embedding(self):
        # Initialize the EnergyAdaptor module
        energy_adaptor = EnergyAdaptor(
            channels_in=self.channels_in,
            channels_hidden=self.channels_hidden,
            channels_out=self.channels_out,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            leaky_relu_slope=self.leaky_relu_slope,
            emb_kernel_size=self.emb_kernel_size,
        )

        # Test add_energy_embedding method
        x_with_energy, energy_pred = energy_adaptor.add_energy_embedding(
            x=self.inputs,
            mask=self.mask,
        )

        # Check shapes of output tensors
        self.assertEqual(x_with_energy.shape, self.inputs.shape)
        self.assertEqual(energy_pred.shape, (self.batch_size, 1, self.seq_length))

if __name__ == "__main__":
    unittest.main()
