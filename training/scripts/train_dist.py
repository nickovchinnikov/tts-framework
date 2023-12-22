import os
from typing import Any

import torch

from lightning.pytorch import Trainer
# from lightning.pytorch.accelerators import find_usable_cuda_devices
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import Dataset

from training.modules import AcousticDataModule, AcousticModule, VocoderModule


class PreprocessedDataset(Dataset):
    r"""A PyTorch Dataset for holding preprocessed data.

    Attributes
        data (list): The preprocessed data.
    """

    def __init__(self, data: Any):
        r"""Initialize the PreprocessedDataset.

        Args:
            data (list): The preprocessed data.
        """
        self.data = data

    def __getitem__(self, index: int):
        r"""Get the data at the given index.

        Args:
            index (int): The index of the data to get.

        Returns:
            The data at the given index.
        """
        return self.data[index]

    def __len__(self):
        r"""Get the number of data in the dataset.

        Returns
            The number of data in the dataset.
        """
        return len(self.data)

# print("usable_cuda_devices: ", find_usable_cuda_devices())

default_root_dir="logs/acoustic"

ckpt_acoustic="./acoustic_epoch8.ckpt"

# TODO: changes in LR is maybe not the best way to do this...
ckpt_acoustic_fixedlr="./acoustic_epoch8.ckpt_fixedlr.ckpt"

# Change the LR in the checkpoint
ckpt_acoustic_loaded = torch.load(ckpt_acoustic)

latest_lr = ckpt_acoustic_loaded["optimizer_states"][0]['param_groups'][0]["lr"]
ckpt_acoustic_loaded["optimizer_states"][0]['param_groups'][0]["lr"] = latest_lr * 2000

# Save the new updated checkpoint
torch.save(ckpt_acoustic_loaded, ckpt_acoustic_fixedlr)


ckpt_vocoder="./vocoder.ckpt"

# Control Validation Frequency
# check_val_every_n_epoch=10
# Accumulate gradients
accumulate_grad_batches=5
# SWA learning rate
swa_lrs=1e-2

# Stochastic Weight Averaging (SWA) can make your models generalize
# better at virtually no additional cost.
# This can be used with both non-trained and trained models.
# The SWA procedure smooths the loss landscape thus making it
# harder to end up in a local minimum during optimization.
callbacks = [
    # StochasticWeightAveraging(swa_lrs=swa_lrs),
    # TODO: Add EarlyStopping Callback
]

tensorboard = TensorBoardLogger(save_dir=default_root_dir)

trainer = Trainer(
    accelerator="cuda",
    devices=-1,
    strategy="ddp_find_unused_parameters_true",
    logger=tensorboard,
    # Save checkpoints to the `default_root_dir` directory
    default_root_dir=default_root_dir,
    # check_val_every_n_epoch=check_val_every_n_epoch,
    accumulate_grad_batches=accumulate_grad_batches,
    max_epochs=-1,
    # callbacks=callbacks,
)

# Load the pretrained weights for the vocoder
vocoder_module = VocoderModule.load_from_checkpoint(
    ckpt_vocoder,
)

module = AcousticModule.load_from_checkpoint(
    ckpt_acoustic,
    vocoder_module=vocoder_module,
)

# datamodule = AcousticDataModule(batch_size=module.train_config.batch_size)

# Create a Tuner
# tuner = Tuner(trainer)

# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
# tuner.lr_find(module)

# tuner.scale_batch_size(module, datamodule=datamodule)

# vocoder_module = VocoderModule()
# module = AcousticModule()

train_dataloader = module.train_dataloader()

trainer.fit(
    model=module,
    train_dataloaders=train_dataloader,
    # Resume training states from the checkpoint file
    ckpt_path=ckpt_acoustic,
)