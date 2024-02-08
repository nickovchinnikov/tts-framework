import os
import unittest

from lightning.pytorch import Trainer
import torch

from models.config import VocoderFinetuningConfig, VocoderPretrainingConfig
from models.vocoder.univnet import UnivNet

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestUnivNet(unittest.TestCase):
    def test_optim_finetuning(self):
        module = UnivNet()

        self.assertIsInstance(module.train_config, VocoderPretrainingConfig)

        optimizer_configs = module.configure_optimizers()

        for optimizer_config in optimizer_configs:
            optimizer = optimizer_config["optimizer"]
            lr_scheduler = optimizer_config["lr_scheduler"]

            # Test the optimizer
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_finetuning(self):
        module = UnivNet(fine_tuning=True)

        self.assertIsInstance(module.train_config, VocoderFinetuningConfig)

    # def test_train_step(self):
    #     trainer = Trainer(
    #         # Save checkpoints to the `default_root_dir` directory
    #         default_root_dir="checkpoints/vocoder",
    #         limit_train_batches=2,
    #         max_epochs=1,
    #         # accelerator="cuda",
    #     )

    #     # Load the pretrained weights
    #     # NOTE: this is the path to the checkpoint in the repo
    #     # It works only for version 0.1.0 checkpoint
    #     # This code will be removed in the future!
    #     checkpoint_path = "models/checkpoints/assets/v0.1.0/vocoder_pretrained.pt"

    #     module = VocoderModule(checkpoint_path_v1=checkpoint_path)

    #     train_dataloader = module.train_dataloader()

    #     result = trainer.fit(model=module, train_dataloaders=train_dataloader)

    #     self.assertIsNone(result)

    def test_load_from_checkpoint(self):
        try:
            UnivNet.load_from_checkpoint(
                "./checkpoints/vocoder.ckpt",
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

