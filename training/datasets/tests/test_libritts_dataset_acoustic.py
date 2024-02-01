import unittest

import os

import torch
from torch.utils.data import DataLoader


from training.datasets import LibriTTSDatasetAcoustic


class TestLibriTTSDatasetAcoustic(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.lang = "en"
        self.download = False

        self.dataset = LibriTTSDatasetAcoustic(
            root="datasets_cache/LIBRITTS",
            lang=self.lang,
            download=self.download,
        )


    def test_len(self):
        self.assertEqual(len(self.dataset), 33236)


    def test_getitem(self):
        sample = self.dataset[0]
        self.assertEqual(sample["id"], "1034_121119_000001_000001")
        self.assertEqual(sample["speaker"], 1034)

        self.assertEqual(sample["text"].shape, torch.Size([6]))
        self.assertEqual(sample["mel"].shape, torch.Size([100, 58]))
        self.assertEqual(sample["pitch"].shape, torch.Size([58]))
        self.assertEqual(sample["raw_text"], "The Law.")
        self.assertEqual(sample["normalized_text"], "The Law.")
        self.assertFalse(sample["pitch_is_normalized"])
        self.assertEqual(sample["lang"], 3)
        self.assertEqual(sample["attn_prior"].shape, torch.Size([6, 58]))
        self.assertEqual(sample["wav"].shape, torch.Size([1, 14994]))
        self.assertEqual(sample["energy"].shape, torch.Size([1, 957]))

    def test_cache_item(self):
        dataset = LibriTTSDatasetAcoustic(
            cache=True,
        )

        idxs = [0, 1, 1000, 1002, 2010]

        for idx in idxs:
            # Get a sample from the dataset
            sample = dataset[idx]

            cache_subdir_path = os.path.join(dataset.cache_dir, dataset.cache_subdir(idx))
            cache_file = os.path.join(cache_subdir_path, f'{idx}.pt')

            # Check if the data is in the cache
            self.assertTrue(os.path.exists(cache_file))

            # Load the data from the cache file
            cached_sample = torch.load(cache_file)

            # Check if the cached data is the same as the original data
            for key in sample:
                if torch.is_tensor(sample[key]):
                    self.assertTrue(torch.all(sample[key] == cached_sample[key]))
                else:
                    self.assertEqual(sample[key], cached_sample[key])


    def test_collate_fn(self):
        data = [
            self.dataset[0],
            self.dataset[2],
        ]

        # Call the collate_fn method
        result = self.dataset.collate_fn(data)

        # Check the output
        self.assertEqual(len(result), 13)

        # Check that all the batches are the same size
        for batch in result:
            self.assertEqual(len(batch), 2)


    def test_normalize_pitch(self):
        pitches = [
            torch.tensor([100.0, 200.0, 300.0]),
            torch.tensor([150.0, 250.0, 350.0]),
        ]

        result = self.dataset.normalize_pitch(pitches)

        expected_output = (100.0, 350.0, 225.0, 93.54143524169922)

        self.assertEqual(result, expected_output)

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
            # items = batch[0]

            # Check the batch size
            self.assertEqual(len(items), 13)

            for it in items:
                self.assertEqual(len(it), self.batch_size)


if __name__ == "__main__":
    unittest.main()
