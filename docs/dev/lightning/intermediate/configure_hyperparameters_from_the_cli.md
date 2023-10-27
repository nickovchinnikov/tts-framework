## Configure hyperparameters from the CLI (Intermediate)

#### [More details are here: Configure hyperparameters from the CLI (Intermediate)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html)

### LightningCLI requirements

The `LightningCLI` class is designed to significantly ease the implementation of CLIs. To use this class, an additional Python requirement is necessary than the minimal installation of Lightning provides. To enable, either install all extras:

```bash
pip install "lightning[pytorch-extra]"
```

or if only interested in `LightningCLI`, just install `jsonargparse`:


```bash
pip install "jsonargparse[signatures]"
```

### Implementing a CLI

Implementing a CLI is as simple as instantiating a `LightningCLI` object giving as arguments classes for a `LightningModule` and optionally a `LightningDataModule`:

```python
# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block

```

Now your model can be managed via the CLI. To see the available commands type:

```bash
python main.py --help
```

which prints out:

```bash
usage: main.py [-h] [-c CONFIG] [--print_config [={comments,skip_null,skip_default}+]]
        {fit,validate,test,predict} ...

Lightning Trainer command line tool

optional arguments:
-h, --help            Show this help message and exit.
-c CONFIG, --config CONFIG
                        Path to a configuration file in json or yaml format.
--print_config [={comments,skip_null,skip_default}+]
                        Print configuration and exit.

subcommands:
For more details of each subcommand add it as argument followed by --help.

{fit,validate,test,predict}
    fit                 Runs the full optimization routine.
    validate            Perform one evaluation epoch over the validation set.
    test                Perform one evaluation epoch over the test set.
    predict             Run inference on your data.
```

The message tells us that we have a few available subcommands:

```bash
python main.py [subcommand]
```

which you can use depending on your use case:

```bash
python main.py fit
python main.py validate
python main.py test
python main.py predict
```

### Train a model with the CLI

To train a model, use the `fit` subcommand:

```bash
python main.py fit
```

View all available options with the `--help` argument given after the subcommand:

```bash
python main.py fit --help

usage: main.py [options] fit [-h] [-c CONFIG]
                            [--seed_everything SEED_EVERYTHING] [--trainer CONFIG]
                            ...
                            [--ckpt_path CKPT_PATH]
    --trainer.logger LOGGER

optional arguments:
<class '__main__.DemoModel'>:
    --model.out_dim OUT_DIM
                            (type: int, default: 10)
    --model.learning_rate LEARNING_RATE
                            (type: float, default: 0.02)
<class 'lightning.pytorch.demos.boring_classes.BoringDataModule'>:
--data CONFIG         Path to a configuration file.
--data.data_dir DATA_DIR
                        (type: str, default: ./)
```

With the Lightning CLI enabled, you can now change the parameters without touching your code:

```bash
# change the learning_rate
python main.py fit --model.learning_rate 0.1

# change the output dimensions also
python main.py fit --model.out_dim 10 --model.learning_rate 0.1

# change trainer and data arguments too
python main.py fit --model.out_dim 2 --model.learning_rate 0.1 --data.data_dir '~/' --trainer.logger False
```

> The options that become available in the CLI are the `__init__` parameters of the `LightningModule` and `LightningDataModule` classes. Thus, to make hyperparameters configurable, just add them to your class’s `__init__`. It is highly recommended that these parameters are described in the docstring so that the CLI shows them in the help. Also, the parameters should have accurate type hints so that the CLI can fail early and give understandable error messages when incorrect values are given.

### Why mix models and datasets

Lightning projects usually begin with one model and one dataset. As the project grows in complexity and you introduce more models and more datasets, it becomes desirable to mix any model with any dataset directly from the command line without changing your code.

```bash
# Mix and match anything
python main.py fit --model=GAN --data=MNIST
python main.py fit --model=Transformer --data=MNIST
```

`LightningCLI` makes this very simple. Otherwise, this kind of configuration requires a significant amount of boilerplate that often looks like this:

```python
# choose model
if args.model == "gan":
    model = GAN(args.feat_dim)
elif args.model == "transformer":
    model = Transformer(args.feat_dim)
...

# choose datamodule
if args.data == "MNIST":
    datamodule = MNIST()
elif args.data == "imagenet":
    datamodule = Imagenet()
...

# mix them!
trainer.fit(model, datamodule)
```

NOTE: It is highly recommended that you avoid writing this kind of boilerplate and use `LightningCLI` instead.

### Multiple LightningModules

To support multiple models, when instantiating `LightningCLI` omit the `model_class` parameter:

```python
# main.py
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class Model1(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model1", "⚡")
        return super().configure_optimizers()


class Model2(DemoModel):
    def configure_optimizers(self):
        print("⚡", "using Model2", "⚡")
        return super().configure_optimizers()


cli = LightningCLI(datamodule_class=BoringDataModule)
```

Now you can choose between any model from the CLI:

```bash
# use Model1
python main.py fit --model Model1

# use Model2
python main.py fit --model Model2

```

> Note: Instead of omitting the model_class parameter, you can give a base class and `subclass_mode_model=True`. This will make the CLI only accept models which are a subclass of the given base class.

### Multiple LightningDataModules

To support multiple data modules, when instantiating `LightningCLI` omit the `datamodule_class` parameter:

```python
# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class FakeDataset1(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset1", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


class FakeDataset2(BoringDataModule):
    def train_dataloader(self):
        print("⚡", "using FakeDataset2", "⚡")
        return torch.utils.data.DataLoader(self.random_train)


cli = LightningCLI(DemoModel)
```

Now you can choose between any dataset at runtime:

```bash
# use Model1
python main.py fit --data FakeDataset1

# use Model2
python main.py fit --data FakeDataset2
```

> Instead of omitting the `datamodule_class` parameter, you can give a base class and `subclass_mode_data=True`. This will make the CLI only accept data modules that are a subclass of the given base class.

### Multiple optimizers

Standard optimizers from `torch.optim` work out of the box:

```bash
python main.py fit --optimizer AdamW
```

If the optimizer you want needs other arguments, add them via the CLI (no need to change your code)!

```bash
python main.py fit --optimizer SGD --optimizer.lr=0.01
```

Furthermore, any custom subclass of `torch.optim.Optimizer` can be used as an optimizer:

```python
# main.py
import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule


class LitAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using LitAdam", "⚡")
        super().step(closure)


class FancyAdam(torch.optim.Adam):
    def step(self, closure):
        print("⚡", "using FancyAdam", "⚡")
        super().step(closure)


cli = LightningCLI(DemoModel, BoringDataModule)
```

Now you can choose between any optimizer at runtime:

```bash
# use LitAdam
python main.py fit --optimizer LitAdam

# use FancyAdam
python main.py fit --optimizer FancyAdam
```

Maybe it's an overhead. I'm not sure about this approach.

#### [More details are here: Configure hyperparameters from the CLI (Intermediate)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_intermediate_2.html)
