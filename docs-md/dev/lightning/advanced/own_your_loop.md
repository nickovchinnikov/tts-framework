## Customize training loop

#### [More details are here: Own your loop (advanced)](https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html)

### Manual Optimization

For advanced research topics like reinforcement learning, sparse coding, or GAN research, it may be desirable to manually manage the optimization process, especially when dealing with multiple optimizers at the same time.

In this mode, Lightning will handle only accelerator, precision and strategy logic.
The users are left with `optimizer.zero_grad()`, gradient accumulation, optimizer toggling, etc.

To manually optimize, do the following:

* Set `self.automatic_optimization=False` in your `LightningModule`'s `__init__`.
* Use the following functions and call them manually:
    - `self.optimizers()` to access your optimizers (one or multiple)
    - `optimizer.zero_grad()` to clear the gradients from the previous training step
    - `self.manual_backward(loss)` instead of `loss.backward()`
    - `optimizer.step()` to update your model parameters
    - `self.toggle_optimizer()` and `self.untoggle_optimizer()` if needed

Here is a minimal example of manual optimization.

```python
from lightning.pytorch import LightningModule


class MyModel(LightningModule):
    def __init__(self):
        super().__init__()
        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        loss = self.compute_loss(batch)
        self.manual_backward(loss)
        opt.step()
```

#### Access your Own Optimizer

The provided optimizer is a `LightningOptimizer` object wrapping your own optimizer configured in your `configure_optimizers()`. You can access your own optimizer with `optimizer.optimizer`. However, if you use your own optimizer to perform a step, Lightning won't be able to support accelerators, precision and profiling for you.

```python
class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False
        ...

    def training_step(self, batch, batch_idx):
        optimizer = self.optimizers()

        # `optimizer` is a `LightningOptimizer` wrapping the optimizer.
        # To access it, do the following.
        # However, it won't work on TPU, AMP, etc...
        optimizer = optimizer.optimizer
        ...
```

#### Gradient Accumulation

You can accumulate gradients over batches similarly to `accumulate_grad_batches` argument in `Trainer` for automatic optimization. To perform gradient accumulation with one optimizer after every `N` steps, you can do as such.

```python
def __init__(self):
    super().__init__()
    self.automatic_optimization = False


def training_step(self, batch, batch_idx):
    opt = self.optimizers()

    # scale losses by 1/N (for N batches of gradient accumulation)
    loss = self.compute_loss(batch) / N
    self.manual_backward(loss)

    # accumulate gradients of N batches
    if (batch_idx + 1) % N == 0:
        opt.step()
        opt.zero_grad()
```

#### Gradient Clipping

You can clip optimizer gradients during manual optimization similar to passing the `gradient_clip_val` and `gradient_clip_algorithm` argument in Trainer during automatic optimization. To perform gradient clipping with one optimizer with manual optimization, you can do as such.

```python
from lightning.pytorch import LightningModule


class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()

        # compute loss
        loss = self.compute_loss(batch)

        opt.zero_grad()
        self.manual_backward(loss)

        # clip gradients
        self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")

        opt.step()
```

> Note that `configure_gradient_clipping()` won't be called in Manual Optimization. Instead consider using self. `clip_gradients()` manually like in the example above.

Note that `configure_gradient_clipping()` won’t be called in Manual Optimization. Instead consider using self. `clip_gradients()` manually like in the example above.

### Use Multiple Optimizers (like GANs)

Here is an example training a simple GAN with multiple optimizers using manual optimization.

```python
import torch
from torch import Tensor
from lightning.pytorch import LightningModule


class SimpleGAN(LightningModule):
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

    def sample_z(self, n) -> Tensor:
        sample = self._Z.sample((n,))
        return sample

    def sample_G(self, n) -> Tensor:
        z = self.sample_z(n)
        return self.G(z)

    def training_step(self, batch, batch_idx):
        # Implementation follows the PyTorch tutorial:
        # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
        g_opt, d_opt = self.optimizers()

        X, _ = batch
        batch_size = X.shape[0]

        real_label = torch.ones((batch_size, 1), device=self.device)
        fake_label = torch.zeros((batch_size, 1), device=self.device)

        g_X = self.sample_G(batch_size)

        ##########################
        # Optimize Discriminator #
        ##########################
        d_x = self.D(X)
        errD_real = self.criterion(d_x, real_label)

        d_z = self.D(g_X.detach())
        errD_fake = self.criterion(d_z, fake_label)

        errD = errD_real + errD_fake

        d_opt.zero_grad()
        self.manual_backward(errD)
        d_opt.step()

        ######################
        # Optimize Generator #
        ######################
        d_z = self.D(g_X)
        errG = self.criterion(d_z, real_label)

        g_opt.zero_grad()
        self.manual_backward(errG)
        g_opt.step()

        self.log_dict({"g_loss": errG, "d_loss": errD}, prog_bar=True)

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.G.parameters(), lr=1e-5)
        d_opt = torch.optim.Adam(self.D.parameters(), lr=1e-5)
        return g_opt, d_opt
```

### Learning Rate Scheduling

Every optimizer you use can be paired with any Learning Rate Scheduler. Please see the documentation of `configure_optimizers()` for all the available options

You can call `lr_scheduler.step()` at arbitrary intervals. Use `self.lr_schedulers()` in your `LightningModule` to access any learning rate schedulers defined in your `configure_optimizers()`.

#### [More details are here: Own your loop (advanced)](https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html)
