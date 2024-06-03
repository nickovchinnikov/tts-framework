## Track and Visualize Experiments

#### [More details are here: Track and Visualize Experiments](https://lightning.ai/docs/pytorch/stable/visualize/logging_intermediate.html)

### Track audio and other artifacts

To track other artifacts, such as histograms or model topology graphs first select one of the many loggers supported by Lightning

```python
from lightning.pytorch import loggers as pl_loggers

tensorboard = pl_loggers.TensorBoardLogger(save_dir="")
trainer = Trainer(logger=tensorboard)
```

then access the logger’s API directly

```python
def training_step(self):
    tensorboard = self.logger.experiment
    tensorboard.add_image()
    tensorboard.add_histogram(...)
    tensorboard.add_figure(...)
```

#### [More details are here: Track and Visualize Experiments](https://lightning.ai/docs/pytorch/stable/visualize/logging_intermediate.html)

Libs:

#### [mlflow](https://mlflow.org/)

#### [comet](https://www.comet.com/site/)
