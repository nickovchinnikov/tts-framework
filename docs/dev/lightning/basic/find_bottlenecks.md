## Find training loop bottlenecks

#### [More details are here: Find bottlenecks](https://lightning.ai/docs/pytorch/stable/tuning/profiler_basic.html)

The most basic profile measures all the key methods across `Callbacks`, `DataModules` and the `LightningModule` in the training loop.

```python
trainer = Trainer(profiler="simple")
```

Once the `.fit()` function has completed, you’ll see an output like this:

```
FIT Profiler Report

-----------------------------------------------------------------------------------------------
|  Action                                          |  Mean duration (s)     |  Total time (s) |
-----------------------------------------------------------------------------------------------
|  [LightningModule]BoringModel.prepare_data       |  10.0001               |  20.00          |
|  run_training_epoch                              |  6.1558                |  6.1558         |
|  run_training_batch                              |  0.0022506             |  0.015754       |
|  [LightningModule]BoringModel.optimizer_step     |  0.0017477             |  0.012234       |
|  [LightningModule]BoringModel.val_dataloader     |  0.00024388            |  0.00024388     |
|  on_train_batch_start                            |  0.00014637            |  0.0010246      |
|  [LightningModule]BoringModel.teardown           |  2.15e-06              |  2.15e-06       |
|  [LightningModule]BoringModel.on_train_start     |  1.644e-06             |  1.644e-06      |
|  [LightningModule]BoringModel.on_train_end       |  1.516e-06             |  1.516e-06      |
|  [LightningModule]BoringModel.on_fit_end         |  1.426e-06             |  1.426e-06      |
|  [LightningModule]BoringModel.setup              |  1.403e-06             |  1.403e-06      |
|  [LightningModule]BoringModel.on_fit_start       |  1.226e-06             |  1.226e-06      |
-----------------------------------------------------------------------------------------------
```

## Profile the time within every function

```python
trainer = Trainer(profiler="advanced")
```

Once the `.fit()` function has completed, you’ll see an output like this:

```
Profiler Report

Profile stats for: get_train_batch
        4869394 function calls (4863767 primitive calls) in 18.893 seconds
Ordered by: cumulative time
List reduced from 76 to 10 due to restriction <10>
ncalls  tottime  percall  cumtime  percall filename:lineno(function)
3752/1876    0.011    0.000   18.887    0.010 {built-in method builtins.next}
    1876     0.008    0.000   18.877    0.010 dataloader.py:344(__next__)
    1876     0.074    0.000   18.869    0.010 dataloader.py:383(_next_data)
    1875     0.012    0.000   18.721    0.010 fetch.py:42(fetch)
    1875     0.084    0.000   18.290    0.010 fetch.py:44(<listcomp>)
    60000    1.759    0.000   18.206    0.000 mnist.py:80(__getitem__)
    60000    0.267    0.000   13.022    0.000 transforms.py:68(__call__)
    60000    0.182    0.000    7.020    0.000 transforms.py:93(__call__)
    60000    1.651    0.000    6.839    0.000 functional.py:42(to_tensor)
    60000    0.260    0.000    5.734    0.000 transforms.py:167(__call__)
```

If the profiler report becomes too long, you can stream the report to a file:

```python
from lightning.pytorch.profilers import AdvancedProfiler

profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")
trainer = Trainer(profiler=profiler)
```

## Measure accelerator usage

Another helpful technique to detect bottlenecks is to ensure that you’re using the full capacity of your accelerator (GPU/TPU/IPU/HPU). This can be measured with the `DeviceStatsMonitor`:

```python
from lightning.pytorch.callbacks import DeviceStatsMonitor

trainer = Trainer(callbacks=[DeviceStatsMonitor()])
```

CPU metrics will be tracked by default on the CPU accelerator. To enable it for other accelerators set `DeviceStatsMonitor(cpu_stats=True)`. To disable logging CPU metrics, you can specify `DeviceStatsMonitor(cpu_stats=False)`.

## Find bottlenecks in your code (intermediate)

#### [More details are here: Find bottlenecks in your code (intermediate)](https://lightning.ai/docs/pytorch/stable/tuning/profiler_intermediate.html)

### Profile pytorch operations

To understand the cost of each PyTorch operation, use the `PyTorchProfiler` built on top of the PyTorch profiler.

```python
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler()
trainer = Trainer(profiler=profiler)
```

The profiler will generate an output like this:

```bash
Profiler Report

Profile stats for: training_step
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Name                   Self CPU total %  Self CPU total   CPU total %      CPU total        CPU time avg
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
t                      62.10%           1.044ms          62.77%           1.055ms          1.055ms
addmm                  32.32%           543.135us        32.69%           549.362us        549.362us
mse_loss               1.35%            22.657us         3.58%            60.105us         60.105us
mean                   0.22%            3.694us          2.05%            34.523us         34.523us
div_                   0.64%            10.756us         1.90%            32.001us         16.000us
ones_like              0.21%            3.461us          0.81%            13.669us         13.669us
sum_out                0.45%            7.638us          0.74%            12.432us         12.432us
transpose              0.23%            3.786us          0.68%            11.393us         11.393us
as_strided             0.60%            10.060us         0.60%            10.060us         3.353us
to                     0.18%            3.059us          0.44%            7.464us          7.464us
empty_like             0.14%            2.387us          0.41%            6.859us          6.859us
empty_strided          0.38%            6.351us          0.38%            6.351us          3.175us
fill_                  0.28%            4.782us          0.33%            5.566us          2.783us
expand                 0.20%            3.336us          0.28%            4.743us          4.743us
empty                  0.27%            4.456us          0.27%            4.456us          2.228us
copy_                  0.15%            2.526us          0.15%            2.526us          2.526us
broadcast_tensors      0.15%            2.492us          0.15%            2.492us          2.492us
size                   0.06%            0.967us          0.06%            0.967us          0.484us
is_complex             0.06%            0.961us          0.06%            0.961us          0.481us
stride                 0.03%            0.517us          0.03%            0.517us          0.517us
---------------------  ---------------  ---------------  ---------------  ---------------  ---------------
Self CPU time total: 1.681ms
```

### Profile a distributed model

To profile a distributed model, use the `PyTorchProfiler` with the filename argument which will save a report per rank.

```python
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler(filename="perf-logs")
trainer = Trainer(profiler=profiler)
```

### Visualize profiled operations

To visualize the profiled operations, enable `emit_nvtx` in the `PyTorchProfiler`.

```python
from lightning.pytorch.profilers import PyTorchProfiler

profiler = PyTorchProfiler(emit_nvtx=True)
trainer = Trainer(profiler=profiler)
```

Then run as following:

```bash
nvprof --profile-from-start off -o trace_name.prof -- <regular command here>
```

To visualize the profiled operation, you can either use `nvvp`:

```bash
nvvp trace_name.prof
```

or python:

```bash
python -c 'import torch; print(torch.autograd.profiler.load_nvprof("trace_name.prof"))'
```