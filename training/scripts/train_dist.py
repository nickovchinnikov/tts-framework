import os
from typing import Any

import torch

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices # type: ignore
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.utils.data import Dataset

from training.modules import AcousticDataModule, AcousticModule, VocoderModule

print("usable_cuda_devices: ", find_usable_cuda_devices())

# Set the precision of the matrix multiplication to float32 to improve the performance of the training
torch.set_float32_matmul_precision("high")

default_root_dir="../tts-training-bucket/logs_new_training/"

ckpt_acoustic="./checkpoints/epoch=5537-step=615041.ckpt"

ckpt_vocoder="./vocoder.ckpt"

# TODO: changes in LR is maybe not the best way to do this...
# ckpt_acoustic_fixedlr="./acoustic_epoch8.ckpt_fixedlr.ckpt"

# # Change the LR in the checkpoint
# ckpt_acoustic_loaded = torch.load(ckpt_acoustic)

# latest_lr = ckpt_acoustic_loaded["optimizer_states"][0]['param_groups'][0]["lr"]
# ckpt_acoustic_loaded["optimizer_states"][0]['param_groups'][0]["lr"] = latest_lr * 2000

# # Save the new updated checkpoint
# torch.save(ckpt_acoustic_loaded, ckpt_acoustic_fixedlr)

# Control Validation Frequency
# check_val_every_n_epoch=10
# Accumulate gradients
accumulate_grad_batches=5

# SWA learning rate
# swa_lrs=1e-2

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

module = AcousticModule()

train_dataloader = module.train_dataloader()

trainer.fit(
    model=module,
    train_dataloaders=train_dataloader,
    # Resume training states from the checkpoint file
    # ckpt_path=ckpt_acoustic,
)
