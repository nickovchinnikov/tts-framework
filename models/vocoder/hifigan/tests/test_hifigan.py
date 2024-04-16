import unittest

from lightning.pytorch import Trainer

from models.vocoder.hifigan.hifigan import HifiGan


class TestHifiGan(unittest.TestCase):
    def test_train_steps(self):
        default_root_dir = "checkpoints"

        trainer = Trainer(
            # Save checkpoints to the `default_root_dir` directory
            default_root_dir=default_root_dir,
            fast_dev_run=1,
            limit_train_batches=1,
            max_epochs=1,
            accelerator="cpu",
        )

        module = HifiGan(batch_size=2)

        train_dataloader = module.train_dataloader()

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
