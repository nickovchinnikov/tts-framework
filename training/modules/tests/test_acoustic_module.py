import os
import unittest

from pytorch_lightning import Trainer
import torch

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
        trainer = Trainer(
            # Save checkpoints to the `default_root_dir` directory
            default_root_dir="checkpoints/acoustic",
            limit_train_batches=2,
            max_epochs=1,
            accelerator="cuda",
        )

        # Load the pretrained weights
        # NOTE: this is the path to the checkpoint in the repo
        # It works only for version 0.1.0 checkpoint
        # This code will be removed in the future!
        checkpoint_path = "model/checkpoints/assets/v0.1.0/acoustic_pretrained.pt"

        module = AcousticModule(checkpoint_path_v1=checkpoint_path)

        train_dataloader = module.train_dataloader()

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

    def test_forward(self):
        module = AcousticModule.load_from_checkpoint(
            "./checkpoints/am_pitche_stats.ckpt",
        )

        text = torch.tensor([
            2, 42, 14, 44, 22, 50, 21, 10, 42, 27, 24, 36, 19, 16, 42, 32, 20, 4, 42, 19, 37, 16, 19, 28, 32, 4, 45, 21, 21, 22, 50, 37, 14, 39, 50, 21, 30, 37, 44, 42, 18, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ], device=module.pitches_stat.device)
        src_len = torch.tensor([42], device=module.pitches_stat.device)
        speakers = torch.tensor([
            2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436, 2436,
        ], device=module.pitches_stat.device)
        langs = torch.tensor([
            3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
        ], device=module.pitches_stat.device)

        result = module.forward(text, src_len, speakers, langs)

        self.assertIsInstance(result, torch.Tensor)
