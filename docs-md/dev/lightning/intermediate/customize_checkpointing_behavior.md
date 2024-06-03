## Customize checkpointing behavior

#### [More details are here: Customize checkpointing behavior (intermediate)](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html)

### Modify checkpointing behavior

For fine-grained control over checkpointing behavior, use the `ModelCheckpoint` object

```python
from lightning.pytorch.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(dirpath="my/path/", save_top_k=2, monitor="val_loss")
trainer = Trainer(callbacks=[checkpoint_callback])
trainer.fit(model)
checkpoint_callback.best_model_path
```

Any value that has been logged via `self.log` in the `LightningModule` can be monitored.

```python
class LitModel(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        self.log("my_metric", x)


# 'my_metric' is now able to be monitored
checkpoint_callback = ModelCheckpoint(monitor="my_metric")
```

#### [More details are here: Customize checkpointing behavior (intermediate)](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html)