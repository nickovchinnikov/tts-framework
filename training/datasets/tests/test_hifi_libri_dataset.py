from pathlib import Path
import unittest

from torch import Tensor

from training.datasets.hifi_libri_dataset import HifiLibriDataset, HifiLibriItem


class TestHifiLibriDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = HifiLibriDataset(cache_dir="datasets_cache", cache=True)

    def test_init(self):
        self.assertEqual(len(self.dataset.cutset), 327197)

    def test_get_cache_subdir_path(self):
        idx = 1234
        expected_path = Path("/dev/shm") / "cache-hifitts-librittsr" / "2000"
        self.assertEqual(self.dataset.get_cache_subdir_path(idx), expected_path)

    def test_get_cache_file_path(self):
        idx = 1234
        expected_path = (
            Path("/dev/shm") / "cache-hifitts-librittsr" / "2000" / f"{idx}.pt"
        )
        self.assertEqual(self.dataset.get_cache_file_path(idx), expected_path)

    def test_getitem(self):
        # Take the hifi items from the beginning of the dataset
        item = self.dataset[0]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "hifitts")
        # Check that the cache file is created
        cache_file = self.dataset.get_cache_file_path(0)
        self.assertTrue(cache_file.exists())
        # Take the same id again to check if the cache is used
        item = self.dataset[0]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "hifitts")

        item = self.dataset[10]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "hifitts")
        # Check that the cache file is created
        cache_file = self.dataset.get_cache_file_path(10)
        self.assertTrue(cache_file.exists())

        item = self.dataset[20]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "hifitts")

        # Take the libri items from the end of the dataset
        item = self.dataset[len(self.dataset) - 20]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "libritts")
        # Check that the cache file is created
        cache_file = self.dataset.get_cache_file_path(len(self.dataset) - 20)
        self.assertTrue(cache_file.exists())

        item = self.dataset[len(self.dataset) - 10]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "libritts")
        item = self.dataset[len(self.dataset) - 5]
        self.assertIsInstance(item, HifiLibriItem)
        self.assertEqual(item.dataset_type, "libritts")

    def test_collate_fn(self):
        data = [self.dataset[0] for _ in range(10)]
        collated = self.dataset.collate_fn(data)
        self.assertIsInstance(collated, list)
        self.assertIsInstance(collated[0], list)  # ids
        self.assertIsInstance(collated[1], list)  # raw_texts
        self.assertIsInstance(collated[2], Tensor)  # speakers
        self.assertIsInstance(collated[3], Tensor)  # texts
        self.assertIsInstance(collated[4], Tensor)  # src_lens
        self.assertIsInstance(collated[5], Tensor)  # mels
        self.assertIsInstance(collated[6], Tensor)  # pitches
        self.assertIsInstance(collated[7], list)  # pitches_stat
        self.assertIsInstance(collated[8], Tensor)  # mel_lens
        self.assertIsInstance(collated[9], Tensor)  # langs
        self.assertIsInstance(collated[10], Tensor)  # attn_priors
        self.assertIsInstance(collated[11], Tensor)  # wavs
        self.assertIsInstance(collated[12], Tensor)  # energy


if __name__ == "__main__":
    unittest.main()
