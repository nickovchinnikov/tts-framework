## Deploy models into production

#### [More details are here: Deploy models into production](https://lightning.ai/docs/pytorch/stable/common/checkpointing_intermediate.html)

### Compile your model to ONNX

[ONNX](https://pytorch.org/docs/stable/onnx.html) is a package developed by Microsoft to optimize inference. ONNX allows the model to be independent of PyTorch and run on any ONNX Runtime.

To export your model to ONNX format call the `to_onnx()` function on your `LightningModule` with the `filepath` and `input_sample`.

```python
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


# create the model
model = SimpleModel()
filepath = "model.onnx"
input_sample = torch.randn((1, 64))
model.to_onnx(filepath, input_sample, export_params=True)
```

You can also skip passing the input sample if the `example_input_array` property is specified in your `LightningModule`.

```python
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)
        self.example_input_array = torch.randn(7, 64)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


# create the model
model = SimpleModel()
filepath = "model.onnx"
model.to_onnx(filepath, export_params=True)
```

Once you have the exported model, you can run it on your ONNX runtime in the following way:

```python
import onnxruntime

ort_session = onnxruntime.InferenceSession(filepath)
input_name = ort_session.get_inputs()[0].name
ort_inputs = {input_name: np.random.randn(1, 64)}
ort_outs = ort_session.run(None, ort_inputs)
```

### Validate a Model Is Servable

PyTorch Lightning provides a way for you to validate a model can be served even before starting training.

In order to do so, your `LightningModule` needs to subclass the `ServableModule`, implements its hooks and pass a `ServableModuleValidator` callback to the `Trainer`.

Below you can find an example of how the serving of a resnet18 can be validated.

```python
import base64
from dataclasses import dataclass
from io import BytesIO
from os import path
from typing import Dict, Optional

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from lightning.pytorch import LightningDataModule, LightningModule, cli_lightning_logo
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.serve import ServableModule, ServableModuleValidator
from lightning.pytorch.utilities.model_helpers import get_torchvision_model
from PIL import Image as PILImage

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "Datasets")


class LitModule(LightningModule):
    def __init__(self, name: str = "resnet18"):
        super().__init__()
        self.model = get_torchvision_model(name, weights="DEFAULT")
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 10)
        self.criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)


class CIFAR10DataModule(LightningDataModule):
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    def train_dataloader(self, *args, **kwargs):
        trainset = torchvision.datasets.CIFAR10(root=DATASETS_PATH, train=True, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=0)

    def val_dataloader(self, *args, **kwargs):
        valset = torchvision.datasets.CIFAR10(root=DATASETS_PATH, train=False, download=True, transform=self.transform)
        return torch.utils.data.DataLoader(valset, batch_size=2, shuffle=True, num_workers=0)


@dataclass(unsafe_hash=True)
class Image:
    height: Optional[int] = None
    width: Optional[int] = None
    extension: str = "JPEG"
    mode: str = "RGB"
    channel_first: bool = False

    def deserialize(self, data: str) -> torch.Tensor:
        encoded_with_padding = (data + "===").encode("UTF-8")
        img = base64.b64decode(encoded_with_padding)
        buffer = BytesIO(img)
        img = PILImage.open(buffer, mode="r")
        if self.height and self.width:
            img = img.resize((self.width, self.height))
        arr = np.array(img)
        return T.ToTensor()(arr).unsqueeze(0)


class Top1:
    def serialize(self, tensor: torch.Tensor) -> int:
        return torch.nn.functional.softmax(tensor).argmax().item()


class ProductionReadyModel(LitModule, ServableModule):
    def configure_payload(self):
        # 1: Access the train dataloader and load a single sample.
        image, _ = self.trainer.train_dataloader.dataset[0]

        # 2: Convert the image into a PIL Image to bytes and encode it with base64
        pil_image = T.ToPILImage()(image)
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("UTF-8")

        return {"body": {"x": img_str}}

    def configure_serialization(self):
        return {"x": Image(224, 224).deserialize}, {"output": Top1().serialize}

    def serve_step(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {"output": self.model(x)}

    def configure_response(self):
        return {"output": 7}


def cli_main():
    cli = LightningCLI(
        ProductionReadyModel,
        CIFAR10DataModule,
        seed_everything_default=42,
        save_config_kwargs={"overwrite": True},
        run=False,
        trainer_defaults={
            "accelerator": "cpu",
            "callbacks": [ServableModuleValidator()],
            "max_epochs": 1,
            "limit_train_batches": 5,
            "limit_val_batches": 5,
        },
    )
    cli.trainer.fit(cli.model, cli.datamodule)


if __name__ == "__main__":
    cli_lightning_logo()
    cli_main()
```

### Compile your model to TorchScript

#### [More details are here: Compile your model to TorchScript](https://lightning.ai/docs/pytorch/stable/deploy/production_advanced_2.html)

[TorchScript](https://pytorch.org/docs/stable/jit.html) allows you to serialize your models in a way that it can be loaded in non-Python environments. The `LightningModule` has a handy method `to_torchscript()` that returns a scripted module which you can save or directly use.

```python
class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(in_features=64, out_features=4)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))


# create the model
model = SimpleModel()
script = model.to_torchscript()

# save for use in production environment
torch.jit.save(script, "model.pt")
```

It is recommended that you install the latest supported version of `PyTorch` to use this feature without limitations.
Once you have the exported model, you can run it in `PyTorch` or `C++` runtime:

```python
inp = torch.rand(1, 64)
scripted_module = torch.jit.load("model.pt")
output = scripted_module(inp)
```

If you want to script a different method, you can decorate the method with `torch.jit.export()`:

```python
class LitMCdropoutModel(pl.LightningModule):
    def __init__(self, model, mc_iteration):
        super().__init__()
        self.model = model
        self.dropout = nn.Dropout()
        self.mc_iteration = mc_iteration

    @torch.jit.export
    def predict_step(self, batch, batch_idx):
        # enable Monte Carlo Dropout
        self.dropout.train()

        # take average of `self.mc_iteration` iterations
        pred = [self.dropout(self.model(x)).unsqueeze(0) for _ in range(self.mc_iteration)]
        pred = torch.vstack(pred).mean(dim=0)
        return pred


model = LitMCdropoutModel(...)
script = model.to_torchscript(file_path="model.pt", method="script")
```

### Pruning and Quantization

#### [More details are here: Pruning and Quantization](https://lightning.ai/docs/pytorch/stable/deploy/production_advanced_2.html)

Pruning and Quantization are techniques to compress model size for deployment, allowing inference speed up and energy saving without significant accuracy losses.

Pruning has been shown to achieve significant efficiency improvements while minimizing the drop in model performance (prediction quality). Model pruning is recommended for cloud endpoints, deploying models on edge devices, or mobile inference (among others).

To enable pruning during training in Lightning, simply pass in the `ModelPruning` callback to the Lightning Trainer. PyTorch's native pruning implementation is used under the hood.

This callback supports multiple pruning functions: pass any `torch.nn.utils.prune` function as a string to select which weights to prune (`random_unstructured`, `RandomStructured`, etc) or implement your own by subclassing `BasePruningMethod`.

```python
from lightning.pytorch.callbacks import ModelPruning

# set the amount to be the fraction of parameters to prune
trainer = Trainer(callbacks=[ModelPruning("l1_unstructured", amount=0.5)])
```

### Post-training Quantization

Most deep learning applications are using 32-bits of floating-point precision for inference. But low precision data types, especially INT8, are attracting more attention due to significant performance margin. One of the essential concerns of adopting low precision is how to easily mitigate the possible accuracy loss and reach predefined accuracy requirements.

Intel® Neural Compressor, is an open-source Python library that runs on Intel CPUs and GPUs, which could address the aforementioned concern by extending the PyTorch Lightning model with accuracy-driven automatic quantization tuning strategies to help users quickly find out the best-quantized model on Intel hardware. It also supports multiple popular network compression technologies such as sparse, pruning, and knowledge distillation.

#### [More details are here: Post-training Quantization](https://lightning.ai/docs/pytorch/stable/advanced/post_training_quantization.html)


Installation:

```bash
# Install stable basic version from pip
pip install neural-compressor
# Or install stable full version from pip (including GUI)
pip install neural-compressor-full
```

Or from conda:

```bash
# install stable basic version from from conda
conda install opencv-python-headless -c fastai
conda install neural-compressor -c conda-forge -c intel
```

#### Accuracy-driven quantization config

```python
from neural_compressor.config import PostTrainingQuantConfig, TuningCriterion, AccuracyCriterion

accuracy_criterion = AccuracyCriterion(tolerable_loss=0.01)
tuning_criterion = TuningCriterion(max_trials=600)
conf = PostTrainingQuantConfig(
    approach="static", backend="default", tuning_criterion=tuning_criterion, accuracy_criterion=accuracy_criterion
)
```

#### Quantize the model

```python
from neural_compressor.quantization import fit

q_model = fit(model=model.model, conf=conf, calib_dataloader=val_dataloader(), eval_func=eval_func)
```
