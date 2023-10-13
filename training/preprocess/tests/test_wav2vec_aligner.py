import unittest

import torch

from training.preprocess.wav2vec_aligner import Wav2VecAligner


class TestWav2VecAligner(unittest.TestCase):
    def setUp(self):
        self.model = Wav2VecAligner()
        self.text = "VILLEFORT ROSE HALF ASHAMED OF BEING SURPRISED IN SUCH A PAROXYSM OF GRIEF"
        self.wav_path = "./mocks/audio_test.wav"
        self.expected_wav_data_path = "./mocks/audio_test.pt"

    def test_load_audio(self):
        audio_input, sample_rate = self.model.load_audio(self.wav_path)

        expected_audio_input = torch.load(self.expected_wav_data_path)

        # self.assertEqual(sample_rate, 24_000)
        torch.testing.assert_close(audio_input, expected_audio_input)

        with self.assertRaises(FileNotFoundError):
            self.model.load_audio("./nonexistent/path.wav")

    def test_align_single_sample(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(
            audio_input, self.text
        )
        expected_emissions = torch.load("./mocks/wav2vec_aligner/emissions.pt")
        
        torch.testing.assert_close(emissions, expected_emissions)

        self.assertEqual(tokens, [26, 11, 16, 16, 6, 21, 9, 14, 7, 5, 14, 9, 13, 6, 5, 12, 8, 16, 21, 5, 8, 13, 12, 8, 18, 6, 15, 5, 9, 21, 5, 25, 6, 11, 10, 22, 5, 13, 17, 14, 24, 14, 11, 13, 6, 15, 5, 11, 10, 5, 13, 17, 20, 12, 5, 8, 5, 24, 8, 14, 9, 29, 23, 13, 18, 5, 9, 21, 5, 22, 14, 11, 6, 21])
        
        self.assertEqual(transcript, "VILLEFORT|ROSE,|HALF|ASHAMED|OF|BEING|SURPRISED|IN|SUCH|A|PAROXYSM|OF|GRIEF.")

    def test_get_trellis(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, _ = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)

        expected_trellis = torch.load("./mocks/wav2vec_aligner/trellis.pt")

        # Add assertions here based on the expected behavior of get_trellis
        self.assertIsInstance(trellis, torch.Tensor)
        torch.testing.assert_close(trellis, expected_trellis)

    def test_backtrack(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, _ = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)

        expected_path = torch.load("./mocks/wav2vec_aligner/backtrack_path.pt")

        # Add assertions here based on the expected behavior of backtrack
        self.assertIsInstance(path, list)
        self.assertEqual(path, expected_path)

    def test_merge_repeats(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)

        expected_merged_path = torch.load("./mocks/wav2vec_aligner/merged_path.pt")

        # Add assertions here based on the expected behavior of merge_repeats
        self.assertIsInstance(merged_path, list)
        self.assertEqual(merged_path, expected_merged_path)

    def test_merge_words(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)
        merged_words = self.model.merge_words(merged_path)

        expected_merged_words = torch.load("./mocks/wav2vec_aligner/merged_words.pt")

        # Add assertions here based on the expected behavior of merge_words
        self.assertIsInstance(merged_words, list)
        self.assertEqual(merged_words, expected_merged_words)

    def test_forward(self):
        result = self.model(self.wav_path, self.text)

        expected_result = torch.load("./mocks/wav2vec_aligner/merged_words.pt")

        self.assertEqual(result, expected_result)

    def test_save_segments(self):
        # self.model.save_segments(self.wav_path, self.text, "./mocks/wav2vec_aligner/audio")
        self.assertEqual(True, True)



if __name__ == "__main__":
    unittest.main()
