import os

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices # type: ignore
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner

from training.datasets import PreprocessedDataset
from training.modules import AcousticDataModule, AcousticModule, VocoderModule

print("usable_cuda_devices: ", find_usable_cuda_devices())

gcp_logs_bucket = "gs://tts-training-bucket/logs_version12"

default_root_dir="logs/acoustic"

ckpt_acoustic="./logs/acoustic/lightning_logs/version_10/checkpoints/epoch=436-step=107686.ckpt"
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

module = AcousticModule(
    vocoder_module=vocoder_module,
)

# module = AcousticModule.load_from_checkpoint(
#     ckpt_acoustic,
#     vocoder_module=vocoder_module,
# )

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
    ckpt_path=ckpt_acoustic,
    train_dataloaders=train_dataloader,
)
