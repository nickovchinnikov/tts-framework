import unittest

from dp.phonemizer import Phonemizer
import torch
from torch.utils.data import DataLoader

from training.datasets.libritts_dataset import LibriTTSDataset


class TestLibriTTSDataset(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.phonemizer = Phonemizer.from_checkpoint("checkpoints/en_us_cmudict_ipa_forward.pt",)
        self.processing_lang_type = "english_only"
        self.sort = False
        self.drop_last = False
        self.download = False

        self.dataset = LibriTTSDataset(
            root="datasets/LIBRITTS",
            batch_size=self.batch_size,
            phonemizer=self.phonemizer,
            processing_lang_type=self.processing_lang_type,
            sort=self.sort,
            drop_last=self.drop_last,
            download=self.download,
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), 33236)

    def test_getitem(self):
        sample = self.dataset[0]
        self.assertEqual(sample["id"], '1034_121119_000001_000001')
        self.assertEqual(sample["speaker"], 1034)

        # Load the expected output from file
        expected_output = torch.load("mocks/libritts_dataset_0_sample.pt")

        # Check the output
        torch.testing.assert_close(sample["text"], expected_output["text"])
        torch.testing.assert_close(sample["mel"], expected_output["mel"])
        torch.testing.assert_close(sample["pitch"], expected_output["pitch"])

        self.assertEqual(sample["raw_text"], expected_output["raw_text"])
        self.assertEqual(sample["normalized_text"], expected_output["normalized_text"])
        self.assertEqual(sample["pitch_is_normalized"], expected_output["pitch_is_normalized"])
        self.assertEqual(sample["lang"], expected_output["lang"])

        torch.testing.assert_close(sample["attn_prior"], expected_output["attn_prior"])
        torch.testing.assert_close(sample["wav"], expected_output["wav"])

    def test_reprocess(self):
        # Mock the data and indices
        data = [
            self.dataset[0],
            # TODO: Too long record, you need to check this, maybe we can increate the max length!
            # self.dataset[1],
            self.dataset[2],
        ]
        idxs = [0, 1]

        # Call the reprocess method
        result = self.dataset.collate_preprocess(data, idxs)

        # Load the expected output from file
        expected_output = torch.load("mocks/libritts_dataset_test_reprocess_result.pt")

        # Check the output
        self.assertEqual(len(result), 12)
        self.assertEqual(result[0], expected_output[0])
        self.assertEqual(result[1], expected_output[1])
        torch.testing.assert_close(result[2], expected_output[2])
        torch.testing.assert_close(result[3], expected_output[3])
        torch.testing.assert_close(result[4], expected_output[4])
        torch.testing.assert_close(result[5], expected_output[5])
        torch.testing.assert_close(result[6], expected_output[6])
        self.assertEqual(result[7], expected_output[7])
        torch.testing.assert_close(result[8], expected_output[8])
        torch.testing.assert_close(result[9], expected_output[9])
        torch.testing.assert_close(result[10], expected_output[10])
        torch.testing.assert_close(result[11], expected_output[11])

    def test_collate_fn(self):
        data = [
            self.dataset[0],
            self.dataset[2],
        ]

        # Call the collate_fn method
        result = self.dataset.collate_fn(data)

        # Check the output
        self.assertEqual(len(result), 1)

        # Check that all the batches are the same size
        for batch in result[0]:
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
            # collate_fn=self.dataset.collate_fn,
        )

        iter_dataloader = iter(dataloader)

        # Iterate over the DataLoader and check the output
        for i, batch in enumerate([next(iter_dataloader), next(iter_dataloader)]):
            # Check the batch size
            self.assertEqual(len(batch), 4)

            # Check the shapes of the tensors in the batch
            self.assertEqual(batch[0].shape[0], self.batch_size)
            self.assertEqual(batch[1].shape[0], self.batch_size)
            self.assertEqual(batch[2].shape[0], self.batch_size)
            self.assertEqual(batch[3].shape[0], self.batch_size)

            # Check the types of the tensors in the batch
            self.assertIsInstance(batch[0], torch.LongTensor)
            self.assertIsInstance(batch[1], torch.FloatTensor)
            self.assertIsInstance(batch[2], torch.FloatTensor)
            self.assertIsInstance(batch[3], torch.FloatTensor)

            # Check the shapes of the tensors in the batch
            self.assertEqual(batch[0].shape[1], 128)
            self.assertEqual(batch[1].shape[1], 80)
            self.assertEqual(batch[2].shape[1], 1)
            self.assertEqual(batch[3].shape[1], 1)

            # Check that the tensors are on the correct device
            self.assertEqual(batch[0].device, torch.device("cpu"))
            self.assertEqual(batch[1].device, torch.device("cpu"))
            self.assertEqual(batch[2].device, torch.device("cpu"))
            self.assertEqual(batch[3].device, torch.device("cpu"))

if __name__ == "__main__":
    unittest.main()