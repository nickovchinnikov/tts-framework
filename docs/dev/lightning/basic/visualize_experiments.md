## Track and Visualize Experiments

#### [More details are here: Track and Visualize Experiments](https://lightning.ai/docs/pytorch/stable/visualize/logging_basic.html)

In model development, we track values of interest such as the validation_loss to visualize the learning process for our models. Model development is like driving a car without windows, charts and logs provide the windows to know where to drive the car.

With Lightning, you can visualize virtually anything you can think of: numbers, text, images, audio. Your creativity and imagination are the only limiting factor.

### Track metrics

To track a metric, simply use the `self.log` method available inside the `LightningModule`

```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        value = ...
        self.log("some_value", value)
```

To log multiple metrics at once, use `self.log_dict`

```python
values = {"loss": loss, "acc": acc, "metric_n": metric_n}  # add more items if needed
self.log_dict(values)
```

### View in the commandline

To view metrics in the commandline progress bar, set the `prog_bar` argument to True.

```python
self.log(..., prog_bar=True)
```

### View in the browser

By Default, `Lightning` uses `Tensorboard` (if available) and a simple CSV logger otherwise.

```python
# every trainer already has tensorboard enabled by default (if the dependency is available)
trainer = Trainer()
```

To launch the `tensorboard` dashboard run the following command on the commandline.

```python
tensorboard --logdir=lightning_logs/
```

If you’re using a notebook environment, launch Tensorboard with this command

```python
%reload_ext tensorboard
%tensorboard --logdir=lightning_logs/
```

### Accumulate a metric

When `self.log` is called inside the `training_step`, it generates a timeseries showing how the metric behaves over time.

When you call `self.log` inside the `validation_step` and `test_step`, Lightning automatically accumulates the metric and averages it once it’s gone through the whole split (epoch).

```python
def validation_step(self, batch, batch_idx):
    value = batch_idx + 1
    self.log("average_value", value)
```

If you don't want to average you can also choose from `{min,max,sum}` by passing the `reduce_fx` argument.

```python
# default function
self.log(..., reduce_fx="mean")
```

### Configure the saving directory

```python
Trainer(default_root_dir="/your/custom/path")
```
