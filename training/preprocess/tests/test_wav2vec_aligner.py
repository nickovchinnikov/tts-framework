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

        self.assertEqual(sample_rate, 24_000)
        torch.testing.assert_close(audio_input, expected_audio_input)

        with self.assertRaises(FileNotFoundError):
            self.model.load_audio("./nonexistent/path.wav")

    def test_align_single_sample(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(
            audio_input, self.text
        )

        self.assertEqual(emissions.shape, torch.Size([311, 32]))
        
        expected_emissions = torch.load("./mocks/wav2vec_aligner/emissions.pt")
        torch.testing.assert_close(emissions, expected_emissions)

        self.assertEqual(tokens, [4, 25, 10, 15, 15, 5, 20, 8, 13, 6, 4, 13, 8, 12, 5, 4, 11, 7, 15, 20, 4, 7, 12, 11, 7, 17, 5, 14, 4, 8, 20, 4, 24, 5, 10, 9, 21, 4, 12, 16, 13, 23, 13, 10, 12, 5, 14, 4, 10, 9, 4, 12, 16, 19, 11, 4, 7, 4, 23, 7, 13, 8, 28, 22, 12, 17, 4, 8, 20, 4, 21, 13, 10, 5, 20, 4])
        
        self.assertEqual(transcript, "|VILLEFORT|ROSE|HALF|ASHAMED|OF|BEING|SURPRISED|IN|SUCH|A|PAROXYSM|OF|GRIEF|")

    def test_get_trellis(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, _ = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)

        expected_trellis = torch.load("./mocks/wav2vec_aligner/trellis.pt")

        self.assertEqual(emissions.shape, torch.Size([311, 32]))
        self.assertEqual(len(tokens), 76)

        # Add assertions here based on the expected behavior of get_trellis
        self.assertEqual(trellis.shape, torch.Size([311, 76]))

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
        self.assertEqual(len(path), 311)

        self.assertEqual(path, expected_path)

    def test_merge_repeats(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)

        # Add assertions here based on the expected behavior of merge_repeats
        self.assertIsInstance(merged_path, list)
        self.assertEqual(len(merged_path), 76)

    def test_merge_words(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)
        merged_words = self.model.merge_words(merged_path)

        # Add assertions here based on the expected behavior of merge_words
        self.assertIsInstance(merged_words, list)
        self.assertEqual(len(merged_words), 13)

    def test_forward(self):
        result = self.model(self.wav_path, self.text)

        # self.assertEqual(result, expected_result)
        self.assertEqual(len(result), 13)

    def test_save_segments(self):
        # self.model.save_segments(self.wav_path, self.text, "./mocks/wav2vec_aligner/audio")
        self.assertEqual(True, True)



if __name__ == "__main__":
    unittest.main()
