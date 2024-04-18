import os
import unittest

from lightning.pytorch import Trainer
import torch
import torchaudio

from models.delightful_hifi import DelightfulHiFi

checkpoint = "checkpoints/logs_new_training_libri-360-swa_multilingual_conf_epoch=146-step=33516.ckpt"

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TestDelightfulTTS(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_forward(self):
        # Create a dummy Trainer instance
        delightful_checkpoint_path = "checkpoints/epoch=414-step=85176.ckpt"
        hifi_checkpoint_path = "checkpoints/epoch=24-step=14200.ckpt"

        # Load the model
        model = DelightfulHiFi(
            delightful_checkpoint_path=delightful_checkpoint_path,
            hifi_checkpoint_path=hifi_checkpoint_path,
        )

        text_tts = """As the snake shook its head, a deafening shout behind Harry made both of them jump.
        ‘DUDLEY! MR DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON’T BELIEVE WHAT IT’S DOING!’
        "How did you know it was me?" she asked.
        "My dear Professor, I’ve never seen a cat sit so stiffly."
        "You’d be stiff if you’d been sitting on a brick wall all day," said Professor McGonagall.
        """

        speaker_id_ = torch.tensor([1], device=self.device)
        wav = model.forward(text_tts, speaker_id_)

        self.assertIsInstance(wav, torch.Tensor)
