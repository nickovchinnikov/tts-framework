import os
import unittest

from pytorch_lightning import Trainer

from training.modules.acoustic_trainer import AcousticTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestTrainAcousticModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        trainer = Trainer(
            default_root_dir="checkpoints/acoustic",
            limit_train_batches=2,
            max_epochs=1,
            accelerator="cuda",
        )

        model = AcousticTrainer()
        model.eval()

        train_dataloader = model.train_dataloader()

        # Load the pretrained weights
        checkpoint_path = "model/checkpoints/assets/v0.1.0/acoustic_pretrained.pt"
        self.ckpt_acoustic = model.weights_prepare_v1(checkpoint_path)

        # Init the weights, sussesfully!
        model.load_state_dict(self.ckpt_acoustic["gen"], strict=False)

        # self.model.train()
        result = trainer.fit(model=model, train_dataloaders=train_dataloader)
        self.assertIsNone(result)

    def test_load_from_checkpoint(self):
        try:
            AcousticTrainer.load_from_checkpoint(
                "./checkpoints/acoustic/lightning_logs/version_1/checkpoints/epoch=0-step=2.ckpt",
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

