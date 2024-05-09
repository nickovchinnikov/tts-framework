from dataclasses import asdict
from pathlib import Path
import unittest

from training.datasets import HifiGanDataset


class TestHifiGanDataset(unittest.TestCase):
    def setUp(self):
        self.cache_dir = "datasets_cache"
        self.dataset = HifiGanDataset(cache_dir=self.cache_dir, cache=True)

    def test_len(self):
        # Test that the length of the dataset is correct
        self.assertIsInstance(len(self.dataset), int)

    def test_get_cache_subdir_path(self):
        idx = 1234
        expected_path = Path(self.cache_dir) / "cache-hifigan-dataset" / "2000"
        self.assertEqual(self.dataset.get_cache_subdir_path(idx), expected_path)

    def test_get_cache_file_path(self):
        idx = 1234
        expected_path = (
            Path(self.cache_dir) / "cache-hifigan-dataset" / "2000" / f"{idx}.pt"
        )
        self.assertEqual(self.dataset.get_cache_file_path(idx), expected_path)

    def test_getitem(self):
        # Test that getting an item from the dataset returns a HifiGANItem
        item = self.dataset[0]
        self.assertIsInstance(item, tuple)

    def test_iter(self):
        # Test that the dataset is iterable
        for item in self.dataset:
            self.assertIsInstance(item, tuple)
            break

    def test_cache(self):
        cache_file = self.dataset.get_cache_file_path(0)
        self.assertTrue(cache_file.exists())


if __name__ == "__main__":
    unittest.main()
