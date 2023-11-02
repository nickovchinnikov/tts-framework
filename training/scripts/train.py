from argparse import ArgumentParser

from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

from training.modules import AcousticModule, VocoderModule


def train():
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--default_root_dir", type=str, default="checkpoints/acoustic")
    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--ckpt_acoustic", type=str, default="./checkpoints/am_pitche_stats_with_vocoder.ckpt")
    parser.add_argument("--ckpt_vocoder", type=str, default="./checkpoints/vocoder.ckpt")

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    tensorboard = TensorBoardLogger(save_dir=f"{args.default_root_dir}/logs")

    trainer = Trainer(
        logger=tensorboard,
        # Save checkpoints to the `default_root_dir` directory
        default_root_dir=args.default_root_dir,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
    )

    # Load the pretrained weights for the vocoder
    vocoder_module = VocoderModule.load_from_checkpoint(
        args.ckpt_vocoder,
    )
    module = AcousticModule.load_from_checkpoint(
        args.ckpt_acoustic,
        vocoder_module=vocoder_module,
    )

    train_dataloader = module.train_dataloader()

    trainer.fit(model=module, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    train()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
