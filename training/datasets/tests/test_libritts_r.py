import os
import unittest

from training.datasets.libritts_r import LIBRITTS_R, load_libritts_item


class TestLibriTTS(unittest.TestCase):
    def setUp(self):
        # Set up any necessary values for the tests
        self.fileid = "1061_146197_000015_000000"
        self.path = "datasets_cache/LIBRITTS/LibriTTS/train-clean-360"
        self.ext_audio = ".wav"
        self.ext_original_txt = ".original.txt"
        self.ext_normalized_txt = ".normalized.txt"

    def test_load_libritts_item(self):
        # Test the load_libritts_item function
        waveform, sample_rate, original_text, normalized_text, speaker_id, chapter_id, utterance_id = load_libritts_item(
            self.fileid,
            self.path,
            self.ext_audio,
            self.ext_original_txt,
            self.ext_normalized_txt,
        )

        base_path = os.path.join(
            self.path,
            f"{speaker_id}",
            f"{chapter_id}",
        )

        # Check that the files were created
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    base_path,
                    self.fileid + self.ext_original_txt,
                ),
            ),
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(
                    base_path,
                    self.fileid + self.ext_normalized_txt,
                ),
            ),
        )

    def test_selected_speaker_ids(self):
        # Initialize the dataset with selected speaker IDs
        dataset = LIBRITTS_R(root="datasets_cache/LIBRITTS", url="train-clean-100", selected_speaker_ids=[19, 26])

        # Iterate over the dataset and check the speaker IDs
        for _, _, _, _, speaker_id, _, _ in dataset:
            # Assert that the speaker ID is in the list of selected speaker IDs
            self.assertIn(speaker_id, [19, 26])

    def test_max_audio_length(self):
        # Initialize the dataset with a maximum audio length
        dataset = LIBRITTS_R(
            root="datasets_cache/LIBRITTS",
            url="train-clean-100",
            max_audio_length=3.0,
            selected_speaker_ids=[19, 26],
        )

        # Iterate over the dataset and check the audio lengths
        for waveform, sample_rate, _, _, speaker_id, _, _ in dataset:
            # Get the duration of the waveform in seconds
            duration = waveform.shape[1] / sample_rate

            # Assert that the speaker ID is in the list of selected speaker IDs
            self.assertIn(speaker_id, [19, 26])
            # Assert that the duration is less than or equal to the maximum length
            self.assertLessEqual(duration, 3.0)

    def test_min_audio_length(self):
        # Initialize the dataset with a minimum audio length
        dataset = LIBRITTS_R(
            root="datasets_cache/LIBRITTS",
            url="train-clean-100",
            min_audio_length=30.0,
        )

        # Iterate over the dataset and check the audio lengths
        for waveform, sample_rate, _, _, _, _, _ in dataset:
            # Get the duration of the waveform in seconds
            duration = waveform.shape[1] / sample_rate

            # Assert that the duration is greater than or equal to the minimum length
            self.assertGreaterEqual(duration, 30.0)

    # Add any other assertions you want to make about the return values
    def tearDown(self):
        speaker_id, chapter_id, _, _ = self.fileid.split("_")

        normalized_text_filename = self.fileid + self.ext_normalized_txt
        normalized_text_path = os.path.join(self.path, speaker_id, chapter_id, normalized_text_filename)

        original_text_filename = self.fileid + self.ext_original_txt
        original_text_path = os.path.join(self.path, speaker_id, chapter_id, original_text_filename)

        # Clean up any created files after tests are done
        if os.path.exists(normalized_text_path):
            os.remove(normalized_text_path)
        if os.path.exists(original_text_path):
            os.remove(original_text_path)

if __name__ == "__main__":
    unittest.main()
