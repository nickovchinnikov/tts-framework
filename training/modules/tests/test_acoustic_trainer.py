import os
import unittest

from pytorch_lightning import Trainer

from training.modules.acoustic_trainer import AcousticTrainer

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestTrainAcousticModule(unittest.TestCase):
    def setUp(self):
        pass

    def test_forward(self):
        trainer = Trainer(limit_train_batches=2, max_epochs=1, accelerator="cuda")

        model = AcousticTrainer()
        train_dataloader = model.train_dataloader()

        # Load the pretrained weights
        checkpoint_path = "model/checkpoints/assets/v0.1.0/acoustic_pretrained.pt"
        self.ckpt_acoustic = model.weights_prepare_v1(checkpoint_path)

        # Init the weights, sussesfully!
        model.load_state_dict(self.ckpt_acoustic["gen"], strict=False)

        # self.model.train()
        result = trainer.fit(model=model, train_dataloaders=train_dataloader)
        self.assertIsNone(result)
