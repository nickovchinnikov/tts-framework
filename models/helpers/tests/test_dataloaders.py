import unittest
from unittest.mock import patch

from torch.utils.data import DataLoader

from models.helpers.dataloaders import train_dataloader, train_val_dataloader


class TestDataLoader(unittest.TestCase):
    def test_train_dataloader(self):
        train_loader = train_dataloader(
            batch_size=2,
            num_workers=2,
            cache=False,
            mem_cache=False,
        )

        # Assertions
        self.assertIsInstance(train_loader, DataLoader)

        for batch in train_loader:
            self.assertEqual(len(batch), 13)
            break

    def test_train_val_dataloader(self):
        train_loader, val_loader = train_val_dataloader(
            batch_size=2,
            num_workers=2,
            cache=False,
            mem_cache=False,
        )

        # Assertions
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)

if __name__ == "__main__":
    unittest.main()
