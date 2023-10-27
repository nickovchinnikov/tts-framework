## How to debug Lightning

#### [More details are here: Debug](https://lightning.ai/docs/pytorch/stable/debug/debugging_basic.html)

### Breakpoint (better use debugger)

```python
def function_to_debug():
    x = 2

    # set breakpoint
    import pdb

    pdb.set_trace()
    y = x**2
```

### Run all your model code once quickly

The `fast_dev_run` argument in the trainer runs 5 batch of training, validation, test and prediction data through your trainer to see if there are any bugs:

```python
Trainer(fast_dev_run=True)
```

To change how many batches to use, change the argument to an integer. Here we run 7 batches of each:

```python
Trainer(fast_dev_run=7)
```

### Shorten the epoch length

```python
# use only 10% of training data and 1% of val data
trainer = Trainer(limit_train_batches=0.1, limit_val_batches=0.01)

# use 10 batches of train and 5 batches of val
trainer = Trainer(limit_train_batches=10, limit_val_batches=5)
```

### Sanity Check

Lightning runs 2 steps of validation in the beginning of training. This avoids crashing in the validation loop sometime deep into a lengthy training loop.

```python
trainer = Trainer(num_sanity_val_steps=2)
```

### Print input output layer dimensions

Whenever the `.fit()` function gets called, the Trainer will print the weights summary for the LightningModule.

```python
trainer.fit(...)
```

this generate a table like:

```
  | Name  | Type        | Params
----------------------------------
0 | net   | Sequential  | 132 K
1 | net.0 | Linear      | 131 K
2 | net.1 | BatchNorm1d | 1.0 K
```

To add the child modules to the summary add a `ModelSummary`:

```python
from lightning.pytorch.callbacks import ModelSummary

trainer = Trainer(callbacks=[ModelSummary(max_depth=-1)])
```

To print the model summary if `.fit()` is not called:

```python
from lightning.pytorch.utilities.model_summary import ModelSummary

model = LitModel()
summary = ModelSummary(model, max_depth=-1)
print(summary)
```

To turn off the autosummary use:

```python
Trainer(enable_model_summary=False)
```
