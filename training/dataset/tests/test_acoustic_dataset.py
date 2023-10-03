import unittest

import torch

import numpy as np

from model.config import lang2id
from training.dataset.acoustic_dataset import AcousticDataset


class TestAcousticDataset(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.data_path = "mocks"
        self.assets_path = "mocks"
        self.is_eval = False
        self.sort = False
        self.drop_last = False

        self.dataset = AcousticDataset(
            "metadata.txt",
            self.batch_size,
            self.data_path,
            self.assets_path,
            self.is_eval,
            self.sort,
            self.drop_last,
        )

        self.mel = torch.zeros((80, 100))
        self.pitch = torch.zeros((80, 100))
        self.attn_prior = torch.zeros((100, 80))

        # Create a mock .pt file with the following data
        data = {
            "raw_text": "Hello world",
            "mel": self.mel,
            "pitch": self.pitch,
            "lang": "en",
            "phones": [1, 2, 3, 4, 5],
            "attn_prior": self.attn_prior,
        }

        # Save the dictionary to a .pt file
        torch.save(data, "mocks/data/Alice/0001.pt")
        torch.save(data, "mocks/data/Bob/0002.pt")
        torch.save(data, "mocks/data/Charlie/0003.pt")

    def test_len(self):
        self.assertEqual(len(self.dataset), 3)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertIsInstance(sample, dict)
        self.assertIn("id", sample)
        self.assertIn("speaker_name", sample)

        self.assertIn("speaker", sample)
        self.assertIn("text", sample)

        self.assertIsInstance(sample["text"], torch.Tensor)
        self.assertIn("raw_text", sample)

        self.assertIn("mel", sample)
        self.assertIsInstance(sample["mel"], torch.Tensor)
        # Check that the mel equals to self.mel
        self.assertTrue(torch.equal(sample["mel"], self.mel))

        self.assertIn("pitch", sample)
        self.assertIsInstance(sample["pitch"], torch.Tensor)
        # Check that the pitch equals to self.pitch
        self.assertTrue(torch.equal(sample["pitch"], self.pitch))

        self.assertIn("lang", sample)
        self.assertEqual(sample["lang"], lang2id["en"])
        self.assertIn("attn_prior", sample)
        self.assertIsInstance(sample["attn_prior"], np.ndarray)

    def test_process_meta(self):
        basename, speaker = self.dataset.process_meta("metadata.txt")
        self.assertIsInstance(basename, list)
        self.assertIsInstance(speaker, list)
        self.assertEqual(len(basename), 3)
        self.assertEqual(len(speaker), 3)

    def test_beta_binomial_prior_distribution(self):
        phoneme_count = 10
        mel_count = 20
        prior_dist = self.dataset.beta_binomial_prior_distribution(
            phoneme_count, mel_count
        )
        self.assertIsInstance(prior_dist, np.ndarray)
        self.assertEqual(prior_dist.shape, (mel_count, phoneme_count))

    def test_reprocess(self):
        data = [self.dataset[i] for i in range(len(self.dataset))]
        idxs = [0, 2]
        reprocessed_data = self.dataset.reprocess(data, idxs)
        self.assertIsInstance(reprocessed_data, tuple)
        self.assertEqual(len(reprocessed_data), 11)
        self.assertIsInstance(reprocessed_data[0], list)
        self.assertIsInstance(reprocessed_data[1], list)
        self.assertIsInstance(reprocessed_data[2], np.ndarray)
        self.assertIsInstance(reprocessed_data[3], list)
        self.assertIsInstance(reprocessed_data[4], np.ndarray)
        self.assertIsInstance(reprocessed_data[5], np.ndarray)
        self.assertIsInstance(reprocessed_data[6], np.ndarray)
        self.assertIsInstance(reprocessed_data[7], np.ndarray)
        self.assertIsInstance(reprocessed_data[8], np.ndarray)
        self.assertIsInstance(reprocessed_data[9], np.ndarray)
        self.assertIsInstance(reprocessed_data[10], np.ndarray)

    def test_collate_fn(self):
        data = [self.dataset[i] for i in range(len(self.dataset))]
        batches = self.dataset.collate_fn(data)
        self.assertIsInstance(batches, list)
        self.assertEqual(len(batches), 2)
        self.assertIsInstance(batches[0], tuple)
        self.assertIsInstance(batches[1], tuple)
        self.assertEqual(len(batches[0]), 11)
        self.assertEqual(len(batches[1]), 11)
