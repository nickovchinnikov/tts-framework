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

