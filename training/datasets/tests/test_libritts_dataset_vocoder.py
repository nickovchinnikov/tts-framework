import unittest

import torch
from torch.utils.data import DataLoader

from training.datasets import LibriTTSDatasetVocoder


class TestLibriTTSDatasetAcoustic(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.lang = "en"
        self.sort = False
        self.drop_last = False
        self.download = False

        self.dataset = LibriTTSDatasetVocoder(
            root="datasets_cache/LIBRITTS",
            batch_size=self.batch_size,
            lang=self.lang,
            sort=self.sort,
            drop_last=self.drop_last,
            download=self.download,
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 33236)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertEqual(sample["mel"].shape, torch.Size([100, 64]))
        self.assertEqual(sample["audio"].shape, torch.Size([1, 32703]))
        self.assertEqual(sample["speaker_id"], 1034)

    def test_collate_fn(self):
        data = [
            self.dataset[0],
            self.dataset[2],
        ]

        # Call the collate_fn method
        result = self.dataset.collate_fn(data)

        # Check the output
        self.assertEqual(len(result), 4)

        # Check that all the batches are the same size
        for batch in result:
            self.assertEqual(len(batch), self.batch_size)

    def test_dataloader(self):
        # Create a DataLoader from the dataset
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.dataset.collate_fn,
        )

        iter_dataloader = iter(dataloader)

        # Iterate over the DataLoader and check the output
        for _, items in enumerate([next(iter_dataloader), next(iter_dataloader)]):
            # Check the batch size
            self.assertEqual(len(items), 4)

            for it in items:
                self.assertEqual(len(it), self.batch_size)


if __name__ == "__main__":
    unittest.main()
