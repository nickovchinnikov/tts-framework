from typing import Any

from torch.utils.data import Dataset


class PreprocessedDataset(Dataset):
    r"""A PyTorch Dataset for holding preprocessed data.

    Attributes
        data (list): The preprocessed data.
    """

    def __init__(self, data: Any):
        r"""Initialize the PreprocessedDataset.

        Args:
            data (list): The preprocessed data.
        """
        self.data = data

    def __getitem__(self, index: int):
        r"""Get the data at the given index.

        Args:
            index (int): The index of the data to get.

        Returns:
            The data at the given index.
        """
        return self.data[index]

    def __len__(self):
        r"""Get the number of data in the dataset.

        Returns
            The number of data in the dataset.
        """
        return len(self.data)
