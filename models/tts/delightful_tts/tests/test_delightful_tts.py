import os
import unittest

from lightning.pytorch import Trainer
import torch

from models.config import PreprocessingConfigHifiGAN as PreprocessingConfig
from models.config import PreprocessingConfigUnivNet
from models.tts.delightful_tts.delightful_tts import DelightfulTTS
from models.vocoder.univnet import UnivNet

checkpoint = "checkpoints/logs_44100_tts_80_logs_new3_lightning_logs_version_7_checkpoints_epoch=2450-step=183470.ckpt"

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TestDelightfulTTS(unittest.TestCase):
    def setUp(self):
        # Create a dummy Trainer instance
        self.trainer = Trainer()
        self.preprocessing_config = PreprocessingConfig("multilingual")

    def test_optim_finetuning(self):
        module = DelightfulTTS(preprocess_config=self.preprocessing_config)

        module.trainer = self.trainer

        optimizer_config = module.configure_optimizers()

        optimizer = optimizer_config["optimizer"]
        lr_scheduler = optimizer_config["lr_scheduler"]

        # Test the optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_train_steps(self):
        default_root_dir = "checkpoints/acoustic"
        # checkpoint = "checkpoints/logs_new_training_libri-360-swa_multilingual_conf_epoch=146-step=33516.ckpt"

        trainer = Trainer(
            # Save checkpoints to the `default_root_dir` directory
            default_root_dir=default_root_dir,
            fast_dev_run=1,
            limit_train_batches=1,
            max_epochs=1,
            accelerator="gpu",
        )

        module = DelightfulTTS(
            preprocess_config=self.preprocessing_config,
        )

        train_dataloader = module.train_dataloader(cache=False)

        # automatically restores model, epoch, step, LR schedulers, etc...
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)
        self.assertIsNone(result)

    def test_load_from_new_checkpoint(self):
        try:
            DelightfulTTS.load_from_checkpoint(
                checkpoint,
                strict=False,
                preprocess_config=self.preprocessing_config,
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

    def test_forward(self):
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        module = DelightfulTTS.load_from_checkpoint(
            checkpoint,
            strict=False,
            map_location=device,
            preprocess_config=self.preprocessing_config,
        )
        univnet = UnivNet()
        univnet = univnet.to(device)

        text = """As the snake shook its head, a deafening shout behind Harry made both of them jump.
        'DUDLEY! MR DURSLEY! COME AND LOOK AT THIS SNAKE! YOU WON'T BELIEVE WHAT IT'S DOING!'
        Dudley came waddling towards them as fast as he could.
        ‘Out of the way, you,’ he said, punching Harry in the ribs. Caught by surprise, Harry fell hard on the concrete floor. What came next happened so fast no one saw how it happened – one second, Piers and Dudley were leaning right up close to the glass, the next, they had leapt back with howls of horror."""

        speaker = torch.tensor([2071], device=device)

        mel_spec = module.forward(
            text,
            speaker,
        )

        self.assertIsInstance(mel_spec, torch.Tensor)
