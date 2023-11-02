from argparse import ArgumentParser

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import StochasticWeightAveraging
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner.tuning import Tuner

from training.modules import AcousticModule, VocoderModule


def train():
    parser = ArgumentParser()

    # Trainer arguments
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--default_root_dir", type=str, default="checkpoints/acoustic")

    parser.add_argument("--limit_train_batches", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)

    # Default we use 3 batches to accumulate gradients
    parser.add_argument("--accumulate_grad_batches", type=int, default=3)
    parser.add_argument("--accelerator", type=str, default="cuda")
    parser.add_argument("--ckpt_acoustic", type=str, default="./checkpoints/am_pitche_stats_with_vocoder.ckpt")
    parser.add_argument("--ckpt_vocoder", type=str, default="./checkpoints/vocoder.ckpt")

    # Optimizers
    # FIXME: this is not working, found an errors...
    # Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment
    # Stochastic Weight Averaging (SWA)
    parser.add_argument("--swa", type=bool, default=False)
    # Learning rate finder
    parser.add_argument("--lr_find", type=bool, default=False)
    # Batch size scaling
    parser.add_argument("--batch_size_scaling", type=bool, default=False)

    # Parse the user inputs and defaults (returns a argparse.Namespace)
    args = parser.parse_args()

    tensorboard = TensorBoardLogger(save_dir=args.default_root_dir)

    callbacks = []

    if args.swa:
        callbacks.append(
            # Stochastic Weight Averaging (SWA) can make your models generalize
            # better at virtually no additional cost.
            # This can be used with both non-trained and trained models.
            # The SWA procedure smooths the loss landscape thus making it
            # harder to end up in a local minimum during optimization.
            StochasticWeightAveraging(swa_lrs=1e-2),
        )

    trainer = Trainer(
        logger=tensorboard,
        # Save checkpoints to the `default_root_dir` directory
        default_root_dir=args.default_root_dir,
        limit_train_batches=args.limit_train_batches,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
    )

    # Create a Tuner
    tuner = Tuner(trainer)

    # Load the pretrained weights for the vocoder
    vocoder_module = VocoderModule.load_from_checkpoint(
        args.ckpt_vocoder,
    )
    module = AcousticModule.load_from_checkpoint(
        args.ckpt_acoustic,
        vocoder_module=vocoder_module,
    )

    train_dataloader = module.train_dataloader()

    if args.lr_find:
        # finds learning rate automatically
        # sets hparams.lr or hparams.learning_rate to that learning rate
        tuner.lr_find(module)

    if args.batch_size_scaling:
        # Auto-scale batch size by growing it exponentially (default)
        tuner.scale_batch_size(module, init_val=2, mode="power")

    trainer.fit(model=module, train_dataloaders=train_dataloader)


if __name__ == "__main__":
    train()
    # note: it is good practice to implement the CLI in a function and call it in the main if block
