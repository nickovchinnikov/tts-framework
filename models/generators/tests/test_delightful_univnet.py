import os
import unittest

from lightning.pytorch import Trainer

from models.generators.delightful_univnet import DelightfulUnivnet

checkpoint = "checkpoints/logs_new_training_libri-360_energy_epoch=263-step=45639.ckpt"

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class TestDelightfulUnivnet(unittest.TestCase):
    def test_train_steps(self):
        default_root_dir = "checkpoints/acoustic"

        trainer = Trainer(
            default_root_dir=default_root_dir,
            limit_train_batches=1,
            max_epochs=1,
            accelerator="cpu",
        )

        module = DelightfulUnivnet(batch_size=1, acc_grad_steps=1, swa_steps=1)

        train_dataloader = module.train_dataloader(2, cache=False, mem_cache=False)

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)
        # module.pitches_stat tensor([ 51.6393, 408.3333])
        self.assertIsNone(result)
