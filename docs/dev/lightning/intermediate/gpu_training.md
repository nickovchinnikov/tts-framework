## GPU training

#### [More details are here: GPU training (Basic)](https://lightning.ai/docs/pytorch/stable/accelerators/gpu_basic.html)

The Trainer will run on all available GPUs by default. Make sure you’re running on a machine with at least one GPU. There's no need to specify any NVIDIA flags as Lightning will do it for you.

```python
# run on as many GPUs as available by default
trainer = Trainer(accelerator="auto", devices="auto", strategy="auto")
# equivalent to
trainer = Trainer()

# run on one GPU
trainer = Trainer(accelerator="gpu", devices=1)
# run on multiple GPUs
trainer = Trainer(accelerator="gpu", devices=8)
# choose the number of devices automatically
trainer = Trainer(accelerator="gpu", devices="auto")
```

### Choosing GPU devices

```python
# DEFAULT (int) specifies how many GPUs to use per node
Trainer(accelerator="gpu", devices=k)

# Above is equivalent to
Trainer(accelerator="gpu", devices=list(range(k)))

# Specify which GPUs to use (don't use when running on cluster)
Trainer(accelerator="gpu", devices=[0, 1])

# Equivalent using a string
Trainer(accelerator="gpu", devices="0, 1")

# To use all available GPUs put -1 or '-1'
# equivalent to `list(range(torch.cuda.device_count())) and `"auto"`
Trainer(accelerator="gpu", devices=-1)
```

### Find usable CUDA devices

```python
from lightning.pytorch.accelerators import find_usable_cuda_devices

# Find two GPUs on the system that are not already occupied
trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))

from lightning.fabric.accelerators import find_usable_cuda_devices

# Works with Fabric too
fabric = Fabric(accelerator="cuda", devices=find_usable_cuda_devices(2))
```

