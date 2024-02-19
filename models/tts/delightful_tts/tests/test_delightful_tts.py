import os
import unittest

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torchaudio

from models.tts.delightful_tts import DelightfulTTS

checkpoint = "checkpoints/logs_new_training_libri-360_energy_epoch=263-step=45639.ckpt"

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestDelightfulTTS(unittest.TestCase):
    def test_optim_finetuning(self):
        # Create a dummy Trainer instance
        trainer = Trainer()

        module = DelightfulTTS(fine_tuning=True)

        module.trainer = trainer

        optimizer_config = module.configure_optimizers()

        optimizer = optimizer_config[0]["optimizer"]
        lr_scheduler = optimizer_config[0]["lr_scheduler"]

        # Test the optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_lr_lambda(self):
        # Create a dummy Trainer instance
        trainer = Trainer()

        module = DelightfulTTS()

        module.trainer = trainer

        current_step = 5000

        _, lr_lambda = module.get_lr_lambda()

        # Test the returned function
        self.assertAlmostEqual(lr_lambda(0), 2.0171788261496964e-07, places=10)
        self.assertAlmostEqual(lr_lambda(10), 2.0171788261496965e-06, places=10)
        self.assertAlmostEqual(lr_lambda(100), 2.0171788261496963e-05, places=10)
        self.assertAlmostEqual(lr_lambda(1000), 0.00020171788261496966, places=10)
        self.assertAlmostEqual(lr_lambda(current_step), 0.0007216878364870322, places=10)

    def test_optim_pretraining(self):
        # Create a dummy Trainer instance
        trainer = Trainer()

        module = DelightfulTTS(fine_tuning=False)

        module.trainer = trainer

        optimizer_config = module.configure_optimizers()

        optimizer = optimizer_config[0]["optimizer"]
        lr_scheduler = optimizer_config[0]["lr_scheduler"]

        # Test the optimizer
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR)

    def test_train_steps(self):
        default_root_dir = "checkpoints/acoustic"
        tensorboard = TensorBoardLogger(save_dir=default_root_dir)

        trainer = Trainer(
            logger=tensorboard,
            # Save checkpoints to the `default_root_dir` directory
            default_root_dir=default_root_dir,
            limit_train_batches=1,
            max_epochs=1,
            accelerator="gpu",
            # Precision is set to speed up training
            # precision="bf16-mixed",
            precision="16-mixed",
        )

        module = DelightfulTTS(batch_size=2)

        train_dataloader, _ = module.train_dataloader(2, cache=False, mem_cache=False)

        # automatically restores model, epoch, step, LR schedulers, etc...
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)
        # module.pitches_stat tensor([ 51.6393, 408.3333])
        self.assertIsNone(result)

    def test_load_from_new_checkpoint(self):
        try:
            DelightfulTTS.load_from_checkpoint(
                checkpoint, strict=False,
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

    def test_generate_audio(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        module = DelightfulTTS.load_from_checkpoint(checkpoint, strict=False)

        text = "Hello, this is a test sentence."
        speaker = torch.tensor([100], device=device)

        wav_prediction = module(
            text,
            speaker,
        )

        # Save the audio to a file
        torchaudio.save(        # type: ignore
            "results/output1.wav",
            wav_prediction.unsqueeze(0).detach().cpu(),
            22050,
        )

        self.assertIsInstance(wav_prediction, torch.Tensor)
