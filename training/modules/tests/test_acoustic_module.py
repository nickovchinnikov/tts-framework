import os
import unittest

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
import torch
import torchaudio

from training.modules import AcousticModule

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestTrainAcousticModule(unittest.TestCase):
    def test_optim_finetuning(self):
        module = AcousticModule(fine_tuning=True)

        optimizer_config = module.configure_optimizers()

        optimizer = optimizer_config["optimizer"]
        lr_scheduler = optimizer_config["lr_scheduler"]

        # Test the optimizer
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_lr_lambda(self):
        module = AcousticModule()

        current_step = 5000

        _, lr_lambda = module.get_lr_lambda()

        # Test the returned function
        self.assertAlmostEqual(lr_lambda(0), 2.0171788261496964e-07, places=10)
        self.assertAlmostEqual(lr_lambda(10), 2.0171788261496965e-06, places=10)
        self.assertAlmostEqual(lr_lambda(100), 2.0171788261496963e-05, places=10)
        self.assertAlmostEqual(lr_lambda(1000), 0.00020171788261496966, places=10)
        self.assertAlmostEqual(lr_lambda(current_step), 0.0007216878364870322, places=10)

    def test_optim_pretraining(self):
        module = AcousticModule(fine_tuning=False)

        optimizer_config = module.configure_optimizers()

        optimizer = optimizer_config["optimizer"]
        lr_scheduler = optimizer_config["lr_scheduler"]

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
            limit_train_batches=2,
            max_epochs=1,
            accelerator="cuda",
        )

        # Load the pretrained weights
        # NOTE: this is the path to the checkpoint in the repo
        # It works only for version 0.1.0 checkpoint
        # This code will be removed in the future!
        checkpoint_path = "model/checkpoints/assets/v0.1.0/acoustic_pretrained.pt"

        module = AcousticModule(
            checkpoint_path_v1=checkpoint_path,
        )

        train_dataloader = module.train_dataloader()

        # automatically restores model, epoch, step, LR schedulers, etc...
        # trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)
        # module.pitches_stat tensor([ 51.6393, 408.3333])
        self.assertIsNone(result)

    def test_load_from_new_checkpoint(self):
        try:
            AcousticModule.load_from_checkpoint(
                "./checkpoints/am_pitche_stats.ckpt",
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

    def test_generate_audio(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = "checkpoints/epoch=4796-step=438683.ckpt"
        module = AcousticModule.load_from_checkpoint(checkpoint)

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
