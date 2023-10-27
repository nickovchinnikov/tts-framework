## Early Stopping

#### [More details are here: EarlyStopping](https://lightning.ai/docs/pytorch/stable/common/early_stopping.html)

You can stop and skip the rest of the current epoch early by overriding `on_train_batch_start()` to return `-1` when some condition is met.

### EarlyStopping Callback

The `EarlyStopping` callback can be used to monitor a metric and stop the training when no improvement is observed.

```python
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

# ...
trainer = Trainer(
    # Save checkpoints to the `default_root_dir` directory
    default_root_dir="checkpoints/acoustic",
    limit_train_batches=2,
    max_epochs=1,
    accelerator="cuda",
    # Need to define the criterias
    callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
)
```
