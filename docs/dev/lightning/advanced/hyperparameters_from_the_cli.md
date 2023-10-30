## Configure hyperparameters from the CLI (Advanced)

#### [More details are here: Configure hyperparameters from the CLI (Advanced)](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced.html)

### Run using a config file

To run the CLI using a yaml config, do:

```bash
python main.py fit --config config.yaml
```

Individual arguments can be given to override options in the config file:

```bash
python main.py fit --config config.yaml --trainer.max_epochs 100
```

### Automatic save of config

To ease experiment reporting and reproducibility, by default `LightningCLI` automatically saves the full YAML configuration in the log directory. After multiple fit runs with different hyperparameters, each one will have in its respective log directory a `config.yaml` file. These files can be used to trivially reproduce an experiment, e.g.:

```bash
python main.py fit --config lightning_logs/version_7/config.yaml
```

The automatic saving of the config is done by the special callback `SaveConfigCallback`. This callback is automatically added to the Trainer. To disable the save of the config, instantiate `LightningCLI` with `save_config_callback=None`.

To change the file name of the saved configs to e.g. `name.yaml`, do:

```python
cli = LightningCLI(..., save_config_kwargs={"config_filename": "name.yaml"})
```

It is also possible to extend the `SaveConfigCallback` class, for instance to additionally save the config in a logger. An example of this is:

```python
class LoggerSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


cli = LightningCLI(..., save_config_callback=LoggerSaveConfigCallback)

```

### Prepare a config file for the CLI

The `--help` option of the CLIs can be used to learn which configuration options are available and how to use them. However, writing a config from scratch can be time-consuming and error-prone. To alleviate this, the CLIs have the `--print_config` argument, which prints to stdout the configuration without running the command.

For a CLI implemented as `LightningCLI(DemoModel, BoringDataModule)`, executing:

```bash
python main.py fit --print_config
```

generates a config with all default values like the following:

```yaml
seed_everything: null
trainer:
  logger: true
  ...
model:
  out_dim: 10
  learning_rate: 0.02
data:
  data_dir: ./
ckpt_path: null
```

A standard procedure to run experiments can be:

```bash
# Print a configuration to have as reference
python main.py fit --print_config > config.yaml
# Modify the config to your liking - you can remove all default arguments
nano config.yaml
# Fit your model using the edited configuration
python main.py fit --config config.yaml
```

#### [Customize arguments](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_advanced_2.html)
