import os
import unittest

from pytorch_lightning import Trainer

from training.modules.acoustic_module import AcousticModule

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestTrainAcousticModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
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

        self.assertIsNone(result)

    def test_load_from_checkpoint(self):
        try:
            AcousticModule.load_from_checkpoint(
                "./checkpoints/acoustic/lightning_logs/version_1/checkpoints/epoch=0-step=2.ckpt",
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

