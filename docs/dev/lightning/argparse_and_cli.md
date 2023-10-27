## Hyperparameters from the CLI

#### [More details are here: Configure hyperparameters from the CLI](https://lightning.ai/docs/pytorch/stable/common/hyperparameters.html)

The `ArgumentParser` is a built-in feature in Python that let's you build CLI programs. You can use it to make hyperparameters and other training settings available from the command line:

```python
from argparse import ArgumentParser

parser = ArgumentParser()

# Trainer arguments
parser.add_argument("--devices", type=int, default=2)

# Hyperparameters for the model
parser.add_argument("--layer_1_dim", type=int, default=128)

# Parse the user inputs and defaults (returns a argparse.Namespace)
args = parser.parse_args()

# Use the parsed arguments in your program
trainer = Trainer(devices=args.devices)
model = MyModel(layer_1_dim=args.layer_1_dim)
```

This allows you to call your program like so:

```bash
python trainer.py --layer_1_dim 64 --devices 1
```

### Docs and examples:

#### [argparse — Parser for command-line options, arguments and sub-commands](https://docs.python.org/3/library/argparse.html)

#### [Command-Line Option and Argument Parsing using argparse in Python](https://www.geeksforgeeks.org/command-line-option-and-argument-parsing-using-argparse-in-python/)

#### [Build Command-Line Interfaces With Python's argparse](https://realpython.com/command-line-interfaces-python-argparse/)
