from datetime import datetime
import logging
import os
from pathlib import Path

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner
import torch

from models.tts.delightful_tts.delightful_tts_refined import DelightfulTTS

# Node runk in the cluster
# node_rank = 0
# num_nodes = 1

# # Setup of the training cluster
# os.environ["MASTER_PORT"] = "12355"
# # # Change the IP address to the IP address of the master node
# os.environ["MASTER_ADDR"] = "10.164.0.32"
# os.environ["WORLD_SIZE"] = f"{num_nodes}"
# # # Change the IP address to the IP address of the master node
# os.environ["NODE_RANK"] = f"{node_rank}"

# Get the current date and time
now = datetime.now()

# Format the current date and time as a string
timestamp = now.strftime("%Y%m%d_%H%M%S")

# Create a logger
logger = logging.getLogger("my_logger")

# Set the level of the logger to ERROR
logger.setLevel(logging.ERROR)

# Create a file handler that logs error messages to a file with the current timestamp in its name
handler = logging.FileHandler(f"logs/error_{timestamp}.log")

# Create a formatter and add it to the handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)

print("usable_cuda_devices: ", find_usable_cuda_devices())

# Set the precision of the matrix multiplication to float32 to improve the performance of the training
torch.set_float32_matmul_precision("high")

# logs_10 is the best
# default_root_dir="logs_10"
default_root_dir = "logs_new"

# ckpt_acoustic = (
#     "./logs_11/lightning_logs/version_4/checkpoints/epoch=1639-step=265108.ckpt"
# )

# ckpt_vocoder = "./checkpoints/vocoder.ckpt"

trainer = Trainer(
    accelerator="cuda",
    devices=-1,
    # num_nodes=num_nodes,
    strategy=DDPStrategy(
        gradient_as_bucket_view=True,
        find_unused_parameters=True,
    ),
    # Save checkpoints to the `default_root_dir` directory
    default_root_dir=default_root_dir,
    enable_checkpointing=True,
    accumulate_grad_batches=5,
    max_epochs=-1,
    log_every_n_steps=10,
    gradient_clip_val=0.5,
)

model = DelightfulTTS(batch_size=10)
# model = DelightfulTTS(batch_size=10)
# model = DelightfulTTS.load_from_checkpoint(ckpt_acoustic, strict=False)

# tuner = Tuner(trainer)
# tuner.lr_find(model)

train_dataloader = model.train_dataloader(
    root="/dev/shm/",
    # NOTE: Preload the cached dataset into the RAM
    cache_dir="/dev/shm/",
    cache=True,
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    # Resume training states from the checkpoint file
    # ckpt_path=ckpt_acoustic,
)