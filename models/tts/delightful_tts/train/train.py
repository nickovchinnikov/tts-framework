from datetime import datetime
import logging
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
import torch

from models.tts.delightful_tts import DelightfulTTS

# Node runk in the cluster
node_rank = 0
num_nodes = 2

# Setup of the training cluster
os.environ["MASTER_PORT"] = "12355"
# Change the IP address to the IP address of the master node
os.environ["MASTER_ADDR"] = "10.148.0.6"
os.environ["WORLD_SIZE"] = f"{num_nodes}"
# Change the IP address to the IP address of the master node
os.environ["NODE_RANK"] = f"{node_rank}"

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

default_root_dir="logs"

ckpt_acoustic="./checkpoints/epoch=301-step=124630.ckpt"

ckpt_vocoder="./checkpoints/vocoder.ckpt"

try:
    tensorboard = TensorBoardLogger(
        save_dir=default_root_dir,
    )

    trainer = Trainer(
        accelerator="cuda",
        devices=-1,
        num_nodes=num_nodes,
        strategy=DDPStrategy(
            gradient_as_bucket_view=True,
            find_unused_parameters=True,
        ),
        # strategy="ddp",
        logger=tensorboard,
        # Save checkpoints to the `default_root_dir` directory
        default_root_dir=default_root_dir,
        # accumulate_grad_batches=5,
        max_epochs=-1,
        log_every_n_steps=100,
        # Failed to find the `precision`
        # precision="16-mixed",
        # precision="bf16-mixed",
        # precision="16-mixed",
    )

    model = DelightfulTTS()

    train_dataloader, val_dataloader = model.train_dataloader(
        # NOTE: Preload the cached dataset into the RAM
        cache_dir="/dev/shm/",
        cache=True,
        mem_cache=False,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        # Resume training states from the checkpoint file
        ckpt_path=ckpt_acoustic,
    )

except Exception as e:
    # Log the error message
    logger.error(f"An error occurred: {e}")
    sys.exit(1)
