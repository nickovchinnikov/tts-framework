## Hardware agnostic training (preparation)

#### [More details are here: Hardware agnostic training (preparation)](https://lightning.ai/docs/pytorch/stable/advanced/transfer_learning.html)

### Delete `.cuda()` or `.to()` calls

```python
# before lightning
def forward(self, x):
    x = x.cuda(0)
    layer_1.cuda(0)
    x_hat = layer_1(x)


# after lightning
def forward(self, x):
    x_hat = layer_1(x)
```

Example from the code:

```python
def forward(
    self,
    x: torch.Tensor,
    pitches_range: Tuple[float, float],
    speakers: torch.Tensor,
    langs: torch.Tensor,
    p_control: float = 1.0,
    d_control: float = 1.0,
) -> torch.Tensor:
    # Generate masks for padding positions in the source sequences
    src_mask = tools.get_mask_from_lengths(
        torch.tensor([x.shape[1]], dtype=torch.int64),
    ).to(x.device) # Read the device from the input `x.device`
```

### Synchronize validation and test logging

When running in distributed mode, we have to ensure that the validation and test step logging calls are synchronized across processes. This is done by adding `sync_dist=True` to all `self.log` calls in the validation and test step. This ensures that each GPU worker has the same behaviour when tracking model checkpoints, which is important for later downstream tasks such as testing the best checkpoint across all workers. The `sync_dist` option can also be used in logging calls during the step methods, but be aware that this can lead to significant communication overhead and slow down your training.

```python
def validation_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)


def test_step(self, batch, batch_idx):
    x, y = batch
    logits = self(x)
    loss = self.loss(logits, y)
    # Add sync_dist=True to sync logging across all GPU workers (may have performance impact)
    self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
```

