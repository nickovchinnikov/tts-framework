from datetime import datetime
import logging
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.accelerators import find_usable_cuda_devices  # type: ignore
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.tuner.tuning import Tuner
import torch

from models.config import PreprocessingConfigUnivNet as PreprocessingConfig
from models.tts.delightful_tts.delightful_tts import DelightfulTTS

# Num nodes in the cluster
num_nodes = 1
# Node runk in the cluster
node_rank = 0

os.environ["WORLD_SIZE"] = f"{num_nodes}"
os.environ["NODE_RANK"] = f"{node_rank}"

# IP/Port of the master node
os.environ["MASTER_PORT"] = "12355"
os.environ["MASTER_ADDR"] = "10.148.0.6"

# Create a logger
# Set the level of the logger to ERROR
logger = logging.getLogger("my_logger")
logger.setLevel(logging.ERROR)

# Format the current date and time as a string
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

# Set the logs dir and the checkpoint paths
default_root_dir = "logs"
ckpt_acoustic = "./checkpoints/epoch=301-step=124630.ckpt"
ckpt_vocoder = "./checkpoints/vocoder.ckpt"

try:
    trainer = Trainer(
        accelerator="cuda",
        devices=-1,
        num_nodes=num_nodes,
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

    preprocessing_config = PreprocessingConfig("multilingual")
    model = DelightfulTTS(preprocessing_config)
    # NOTE: Load the model from the checkpoint file
    # In case of loading the model from the checkpoint file, model states will be restored
    # from the checkpoint file but the training states will be reset
    # model = DelightfulTTS.load_from_checkpoint(ckpt_acoustic, strict=False)

    tuner = Tuner(trainer)
    # NOTE: Tune the learning rate of the model if needed
    # tuner.lr_find(model)

    train_dataloader = model.train_dataloader(
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

except Exception as e:
    # Log the error message
    logger.error(f"An error occurred: {e}")
    sys.exit(1)
