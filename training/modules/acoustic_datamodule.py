from lightning.pytorch.core import LightningDataModule
from torch.utils.data import DataLoader

from training.datasets import LibriTTSDatasetAcoustic


class AcousticDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 3):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        self.dataset = LibriTTSDatasetAcoustic()

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=12,
            shuffle=True,
            collate_fn=self.dataset.collate_fn,
        )
