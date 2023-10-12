import unittest

import torch

from training.preprocess.wav2vec_aligner import Wav2VecAligner


class TestWav2VecAligner(unittest.TestCase):
    def setUp(self):
        self.model = Wav2VecAligner()
        self.text = "VILLEFORT ROSE, HALF ASHAMED OF BEING SURPRISED IN SUCH A PAROXYSM OF GRIEF."
        self.wav_path = "./mocks/audio_test.wav"
        self.expected_wav_data_path = "./mocks/audio_test.pt"

    def test_load_audio(self):
        audio_input, sample_rate = self.model.load_audio(self.wav_path)

        expected_audio_input = torch.load(self.expected_wav_data_path)

        self.assertEqual(sample_rate, 24_000)
        torch.testing.assert_close(audio_input, expected_audio_input)

        with self.assertRaises(FileNotFoundError):
            self.model.load_audio("./nonexistent/path.wav")

    def test_text_to_transcript(self):
        transcript = self.model.text_to_transcript(self.text)
        expected_transcript = "VILLEFORT|ROSE,|HALF|ASHAMED|OF|BEING|SURPRISED|IN|SUCH|A|PAROXYSM|OF|GRIEF."
        self.assertEqual(transcript, expected_transcript)

    def test_align_single_sample(self):
        audio_input, _ = self.model.load_audio(self.wav_path)
        emissions, tokens = self.model.align_single_sample(
            audio_input, self.text
        )
        expected_emissions = torch.load("./mocks/wav2vec_aligner/emissions.pt")
        
        torch.testing.assert_close(emissions, expected_emissions)
        self.assertEqual(tokens, [26, 11, 16, 16, 6, 21, 9, 14, 7, 14, 9, 13, 6, 12, 8, 16, 21, 8, 13, 12, 8, 18, 6, 15, 9, 21, 25, 6, 11, 10, 22, 13, 17, 14, 24, 14, 11, 13, 6, 15, 11, 10, 13, 17, 20, 12, 8, 24, 8, 14, 9, 29, 23, 13, 18, 9, 21, 22, 14, 11, 6, 21])

if __name__ == "__main__":
    unittest.main()