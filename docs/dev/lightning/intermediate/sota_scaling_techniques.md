## SOTA scaling techniques

#### [More details are here: Track and Visualize Experiments](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html)

### N-Bit Precision

If you’re looking to run models faster or consume less memory, consider tweaking the precision settings of your models.

Lower precision, such as 16-bit floating-point, requires less memory and enables training and deploying larger models. Higher precision, such as the 64-bit floating-point, can be used for highly sensitive use-cases.

### 16-bit Precision

Use 16-bit mixed precision to speed up training and inference. If your GPUs are [Tensor Core] GPUs, you can expect a ~3x speed improvement.

```python
Trainer(precision="16-mixed")
```

In most cases, mixed precision uses FP16. Supported PyTorch operations automatically run in FP16, saving memory and improving throughput on the supported accelerators. Since computation happens in FP16, which has a very limited “dynamic range”, there is a chance of numerical instability during training. This is handled internally by a dynamic grad scaler which skips invalid steps and adjusts the scaler to ensure subsequent steps fall within a finite range. For more information see the autocast docs.

With true 16-bit precision you can additionally lower your memory consumption by up to half so that you can train and deploy larger models. **However, this setting can sometimes lead to unstable training.**

```python
Trainer(precision="16-true")
```

### 32-bit Precision

32-bit precision is the default used across all models and research. This precision is known to be stable in contrast to lower precision settings.

```python
Trainer(precision="32-true")

# or (legacy)
Trainer(precision="32")

# or (legacy)
Trainer(precision=32)
```

### 64-bit Precision

For certain scientific computations, 64-bit precision enables more accurate models. However, doubling the precision from 32 to 64 bit also doubles the memory requirements.

```python
Trainer(precision="64-true")

# or (legacy)
Trainer(precision="64")

# or (legacy)
Trainer(precision=64)
```

#### [More details are here: Track and Visualize Experiments](https://lightning.ai/docs/pytorch/stable/common/precision_basic.html)


## N-Bit Precision (Intermediate)

#### [More details are here: N-Bit Precision (Intermediate)](https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html)

PyTorch, like most deep learning frameworks, trains on 32-bit floating-point (FP32) arithmetic by default. However, many deep learning models do not require this to reach complete accuracy. By conducting operations in half-precision format while keeping minimum information in single-precision to maintain as much information as possible in crucial areas of the network, mixed precision training delivers significant computational speedup. Switching to mixed precision has resulted in considerable training speedups since the introduction of Tensor Cores in the Volta and Turing architectures. It combines FP32 and lower-bit floating-points (such as FP16) to reduce memory footprint and increase performance during model training and evaluation. It accomplishes this by recognizing the steps that require complete accuracy and employing a 32-bit floating-point for those steps only, while using a 16-bit floating-point for the rest. When compared to complete precision training, mixed precision training delivers all of these benefits while ensuring that no task-specific accuracy is lost.

### BFloat16 Mixed Precision

> BFloat16 may not provide significant speedups or memory improvements or offer better numerical stability. For GPUs, the most significant benefits require Ampere based GPUs or newer, such as A100s or 3090s.

```python
Trainer(accelerator="gpu", devices=1, precision="bf16-mixed")
```

It is also possible to use BFloat16 mixed precision on the CPU, relying on MKLDNN under the hood.

```python
Trainer(precision="bf16-mixed")
```

### True Half Precision

```python
# Select FP16 precision
trainer = Trainer(precision="16-true")
trainer.fit(model)  # model gets cast to torch.float16

# Select BF16 precision
trainer = Trainer(precision="bf16-true")
trainer.fit(model)  # model gets cast to torch.bfloat16
```

> Tip: For faster initialization, you can create model parameters with the desired dtype directly on the device:

```python
trainer = Trainer(precision="bf16-true")

# init the model directly on the device and with parameters in half-precision
with trainer.init_module():
    model = MyModel()

trainer.fit(model)
```

### Efficient initialization (Advanced)

#### [More details are here: Efficient initialization](https://lightning.ai/docs/pytorch/stable/advanced/model_init.html)

#### Half-precision

Instantiating a nn.Module in PyTorch creates all parameters on CPU in float32 precision by default. To speed up initialization, you can force PyTorch to create the model directly on the target device and with the desired precision without changing your model code.

```python
trainer = Trainer(accelerator="cuda", precision="16-true")

with trainer.init_module():
    # models created here will be on GPU and in float16
    model = MyLightningModule()
```

#### Loading checkpoints for inference or finetuning

```python
with trainer.init_module(empty_init=True):
    # creation of the model is fast
    # and depending on the strategy allocates no memory, or uninitialized memory
    model = MyLightningModule.load_from_checkpoint("my/checkpoint/path.ckpt")

trainer.fit(model)
```

### Float8 Mixed Precision via Nvidia’s TransformerEngine

```python
# Select 8bit mixed precision via TransformerEngine, with model weights in bfloat16
trainer = Trainer(precision="transformer-engine")

# Select 8bit mixed precision via TransformerEngine, with model weights in float16
trainer = Trainer(precision="transformer-engine-float16")

# Customize the fp8 recipe or set a different base precision:
from lightning.trainer.plugins import TransformerEnginePrecision

recipe = {"fp8_format": "HYBRID", "amax_history_len": 16, "amax_compute_algo": "max"}
precision = TransformerEnginePrecision(dtype=torch.bfloat16, recipe=recipe)
trainer = Trainer(plugins=precision)
```

> This requires Hopper based GPUs or newer, such the H100.

