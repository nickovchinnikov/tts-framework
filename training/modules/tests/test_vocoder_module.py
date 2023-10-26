import os
import unittest

from pytorch_lightning import Trainer
import torch

from model.config import VocoderFinetuningConfig, VocoderPretrainingConfig
from training.modules import VocoderModule

# NOTE: this is needed to avoid CUDA_LAUNCH_BLOCKING error
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class TestTrainAcousticModule(unittest.TestCase):
    def test_optim_finetuning(self):
        module = VocoderModule()

        self.assertIsInstance(module.train_config, VocoderPretrainingConfig)

        optimizer_configs = module.configure_optimizers()

        for optimizer_config in optimizer_configs:
            optimizer = optimizer_config["optimizer"]
            lr_scheduler = optimizer_config["lr_scheduler"]

            # Test the optimizer
            self.assertIsInstance(optimizer, torch.optim.AdamW)
            self.assertIsInstance(lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)

    def test_finetuning(self):
        module = VocoderModule(fine_tuning=True)

        self.assertIsInstance(module.train_config, VocoderFinetuningConfig)

    def test_weights_loading(self):
        # Check the pretrained weights
        # NOTE: this is the path to the checkpoint in the repo
        # It works only for version 0.1.0 checkpoint
        # This code will be removed in the future!
        checkpoint_path_v1 = "model/checkpoints/assets/v0.1.0/vocoder_pretrained.pt"

        # Initialize the module with a checkpoint
        module = VocoderModule(checkpoint_path_v1=checkpoint_path_v1)

        # Load the checkpoint manually
        checkpoint = torch.load(checkpoint_path_v1)
        expected_generator_weights = checkpoint["generator"]
        expected_discriminator_weights = checkpoint["discriminator"]

        # Get the actual weights of the module
        actual_generator_weights = module.univnet.state_dict()
        actual_discriminator_weights = module.discriminator.state_dict()


        # Check if the weights are the same
        for key in expected_generator_weights:
            assert torch.allclose(
                actual_generator_weights[key],
                expected_generator_weights[key].to(
                    actual_generator_weights[key].device,
                ),
            )

        for key in expected_discriminator_weights:
            assert torch.allclose(
                actual_discriminator_weights[key],
                expected_discriminator_weights[key].to(
                    actual_discriminator_weights[key].device,
                ),
            )

        # Get the actual weights of the optimizers
        univnet_opts, discr_opts = module.configure_optimizers()

        expected_generator_optimizer_weights = checkpoint["optim_g"]
        expected_discriminator_optimizer_weights = checkpoint["optim_d"]

        # Get the actual weights of the optimizers
        actual_generator_optimizer_weights = univnet_opts["optimizer"].state_dict()
        actual_discriminator_optimizer_weights = discr_opts["optimizer"].state_dict()

        def compare_optimizer_state(optimizer1, optimizer2):
            assert len(optimizer1["state"]) == len(optimizer2["state"])
            for state1, state2 in zip(optimizer1["state"].values(), optimizer2["state"].values()):
                for key in state1.keys():
                    if isinstance(state1[key], torch.Tensor):
                        assert torch.allclose(state1[key], state2[key].to(state1[key].device))
                    else:
                        assert state1[key] == state2[key]

        compare_optimizer_state(
            expected_generator_optimizer_weights,
            actual_generator_optimizer_weights,
        )
        compare_optimizer_state(
            expected_discriminator_optimizer_weights,
            actual_discriminator_optimizer_weights,
        )

    def test_train_step(self):
        trainer = Trainer(
            # Save checkpoints to the `default_root_dir` directory
            default_root_dir="checkpoints/vocoder",
            limit_train_batches=2,
            max_epochs=1,
            accelerator="cuda",
        )

        # Load the pretrained weights
        # NOTE: this is the path to the checkpoint in the repo
        # It works only for version 0.1.0 checkpoint
        # This code will be removed in the future!
        checkpoint_path = "model/checkpoints/assets/v0.1.0/vocoder_pretrained.pt"

        module = VocoderModule(checkpoint_path_v1=checkpoint_path)

        train_dataloader = module.train_dataloader()

        result = trainer.fit(model=module, train_dataloaders=train_dataloader)

        self.assertIsNone(result)

    def test_load_from_checkpoint(self):
        try:
            VocoderModule.load_from_checkpoint(
                "./checkpoints/vocoder/lightning_logs/version_17/checkpoints/epoch=0-step=4.ckpt",
            )
        except Exception as e:
            self.fail(f"Loading from checkpoint raised an exception: {e}")

