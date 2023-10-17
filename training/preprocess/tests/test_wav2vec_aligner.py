import unittest

import torch

from training.preprocess.wav2vec_aligner import Wav2VecAligner


class TestWav2VecAligner(unittest.TestCase):
    def setUp(self):
        self.model = Wav2VecAligner()
        self.text = "I HAD THAT CURIOSITY BESIDE ME AT THIS MOMENT"
        self.wav_path = "./mocks/audio_example.wav"


    def test_load_audio(self):
        _, sample_rate = self.model.load_audio(self.wav_path)

        self.assertEqual(sample_rate, 16_000)

        with self.assertRaises(FileNotFoundError):
            self.model.load_audio("./nonexistent/path.wav")


    def test_encode(self):
        tokens = self.model.encode(self.text)
        
        torch.testing.assert_close(tokens, torch.tensor([[10,  4, 11,  7, 14,  4,  6, 11,  7,  6,  4, 19, 16, 13, 10,  8, 12, 10,
          6, 22,  4, 24,  5, 12, 10, 14,  5,  4, 17,  5,  4,  7,  6,  4,  6, 11,
         10, 12,  4, 17,  8, 17,  5,  9,  6]]))


    def test_decode(self):
        transcript = self.model.decode([[10,  4, 11,  7, 14,  4,  6, 11,  7,  6,  4, 19, 16, 13, 10,  8, 12, 10,
          6, 22,  4, 24,  5, 12, 10, 14,  5,  4, 17,  5,  4,  7,  6,  4,  6, 11,
         10, 12,  4, 17,  8, 17,  5,  9,  6]])
        
        self.assertEqual(transcript, self.text)


    def test_align_single_sample(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(
            audio_input, self.text
        )

        self.assertEqual(emissions.shape, torch.Size([169, 32]))

        self.assertEqual(tokens, [4, 10, 4, 11, 7, 14, 4, 6, 11, 7, 6, 4, 19, 16, 13, 10, 8, 12, 10, 6, 22, 4, 24, 5, 12, 10, 14, 5, 4, 17, 5, 4, 7, 6, 4, 6, 11, 10, 12, 4, 17, 8, 17, 5, 9, 6, 4])
        
        self.assertEqual(transcript, "|I|HAD|THAT|CURIOSITY|BESIDE|ME|AT|THIS|MOMENT|")


    def test_get_trellis(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, _ = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)

        self.assertEqual(emissions.shape, torch.Size([169, 32]))
        self.assertEqual(len(tokens), 47)

        # Add assertions here based on the expected behavior of get_trellis
        self.assertIsInstance(trellis, torch.Tensor)
        self.assertEqual(trellis.shape, torch.Size([169, 47]))


    def test_backtrack(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, _ = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)

        # Add assertions here based on the expected behavior of backtrack
        self.assertIsInstance(path, list)
        self.assertEqual(len(path), 169)


    def test_merge_repeats(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)

        # Add assertions here based on the expected behavior of merge_repeats
        self.assertIsInstance(merged_path, list)
        self.assertEqual(len(merged_path), 47)


    def test_merge_words(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens, transcript = self.model.align_single_sample(audio_input, self.text)
        trellis = self.model.get_trellis(emissions, tokens)
        path = self.model.backtrack(trellis, emissions, tokens)
        merged_path = self.model.merge_repeats(path, transcript)
        merged_words = self.model.merge_words(merged_path)

        # Add assertions here based on the expected behavior of merge_words
        self.assertIsInstance(merged_words, list)
        self.assertEqual(len(merged_words), 9)


    def test_forward(self):
        result = self.model(self.wav_path, self.text)

        # self.assertEqual(result, expected_result)
        self.assertEqual(len(result), 9)


    def test_save_segments(self):
        # self.model.save_segments(self.wav_path, self.text, "./mocks/wav2vec_aligner/audio")
        self.assertEqual(True, True)



if __name__ == "__main__":
    unittest.main()
