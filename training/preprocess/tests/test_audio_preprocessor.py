import unittest

import torch

from training.preprocess.audio_processor import AudioProcessor


class TestAudioProcessor(unittest.TestCase):
    def setUp(self):
        self.audio_processor = AudioProcessor()
        self.y = torch.randn(1, 22050)  # 1 second of audio at 22050Hz
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 2048
        self.num_mels = 128
        self.sample_rate = 22050
        self.fmin = 0
        self.fmax = 8000

    def test_name_mel_basis(self):
        spec = torch.randn(1, 1025, 100)
        name = self.audio_processor.name_mel_basis(spec, self.n_fft, self.fmax)
        self.assertEqual(name, f"{self.n_fft}_{self.fmax}_{spec.dtype}_{spec.device}")

    def test_amp_to_db_and_db_to_amp(self):
        magnitudes = torch.abs(torch.randn(1, 100))
        db = self.audio_processor.amp_to_db(magnitudes)
        amp = self.audio_processor.db_to_amp(db)
        self.assertTrue(torch.allclose(magnitudes, amp, atol=1e-4))

    def test_wav_to_spec(self):
        spec = self.audio_processor.wav_to_spec(self.y, self.n_fft, self.hop_length, self.win_length)
        self.assertEqual(spec.shape, (1, self.n_fft // 2 + 1, self.num_mels // 3 + 1))

    def test_wav_to_energy(self):
        energy = self.audio_processor.wav_to_energy(self.y, self.n_fft, self.hop_length, self.win_length)
        self.assertEqual(energy.shape, (1, 1, self.num_mels // 3 + 1))

    def test_spec_to_mel(self):
        spec = torch.randn(1, 1025, 100)
        mel = self.audio_processor.spec_to_mel(
            spec, self.n_fft, self.num_mels, self.sample_rate, self.fmin, self.fmax,
        )
        self.assertEqual(mel.shape, (1, self.num_mels, 100))

    def test_wav_to_mel(self):
        mel = self.audio_processor.wav_to_mel(self.y, self.n_fft, self.num_mels, self.sample_rate, self.hop_length, self.win_length, self.fmin, self.fmax)
        self.assertEqual(mel.shape, (1, self.num_mels, self.num_mels // 3 + 1))


if __name__ == "__main__":
    unittest.main()
