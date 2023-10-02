import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

from pytorch_lightning.core import LightningModule

from model.config import AcousticENModelConfig, PreprocessingConfig
from model.acoustic_model import AcousticModel as AcousticModelBase


class AcousticModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

        preprocess_config = PreprocessingConfig("english_only")
        model_config = AcousticENModelConfig()

        self.model = AcousticModelBase(
            data_path="model/config/",
            preprocess_config=preprocess_config,
            model_config=model_config,
            fine_tuning=True,
            n_speakers=5392,
            # Setup the device, because .to() under the hood of lightning is not working
            device=self.device,
        )

        print(self.model)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        dataset = MNIST(
            os.getcwd(), train=True, download=True, transform=transforms.ToTensor()
        )
        loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)
        return loader
