from typing import List, Optional, Tuple

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler

from training.datasets import LibriTTSDatasetAcoustic


def train_dataloader(
    batch_size: int = 6,
    num_workers: int = 5,
    root: str = "datasets_cache/LIBRITTS",
    cache: bool = True,
    cache_dir: str = "datasets_cache",
    mem_cache: bool = False,
    url: str = "train-clean-360",
    lang: str = "en",
    selected_speaker_ids: Optional[List[int]] = None,
) -> DataLoader:
    r"""Returns the training dataloader, that is using the LibriTTS dataset.

    Args:
        batch_size (int): The batch size.
        num_workers (int): The number of workers.
        root (str): The root directory of the dataset.
        cache (bool): Whether to cache the preprocessed data.
        cache_dir (str): The directory for the cache.
        mem_cache (bool): Whether to use memory cache.
        url (str): The URL of the dataset.
        lang (str): The language of the dataset.
        selected_speaker_ids (Optional[List[int]]): A list of selected speakers.

    Returns:
        DataLoader: The training and validation dataloaders.
    """
    dataset = LibriTTSDatasetAcoustic(
        root=root,
        lang=lang,
        cache=cache,
        cache_dir=cache_dir,
        mem_cache=mem_cache,
        url=url,
        selected_speaker_ids=selected_speaker_ids,
    )

    train_loader = DataLoader(
        dataset,
        # 4x80Gb max 10 sec audio
        # batch_size=20, # self.train_config.batch_size,
        # 4*80Gb max ~20.4 sec audio
        batch_size=batch_size,
        # TODO: find the optimal num_workers
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    return train_loader


def train_val_dataloader(
    batch_size: int = 6,
    num_workers: int = 5,
    root: str = "datasets_cache/LIBRITTS",
    cache: bool = True,
    cache_dir: str = "datasets_cache",
    mem_cache: bool = False,
    url: str = "train-clean-360",
    lang: str = "en",
    validation_split: float = 0.02,  # Percentage of data to use for validation
) -> Tuple[DataLoader, DataLoader]:
    r"""Returns the training dataloader, that is using the LibriTTS dataset.

    Args:
        batch_size (int): The batch size.
        num_workers (int): The number of workers.
        root (str): The root directory of the dataset.
        cache (bool): Whether to cache the preprocessed data.
        cache_dir (str): The directory for the cache.
        mem_cache (bool): Whether to use memory cache.
        url (str): The URL of the dataset.
        lang (str): The language of the dataset.
        validation_split (float): The percentage of data to use for validation.

    Returns:
        Tupple[DataLoader, DataLoader]: The training and validation dataloaders.
    """
    dataset = LibriTTSDatasetAcoustic(
        root=root,
        lang=lang,
        cache=cache,
        cache_dir=cache_dir,
        mem_cache=mem_cache,
        url=url,
    )

    # Split dataset into train and validation
    train_indices, val_indices = train_test_split(
        list(range(len(dataset))),
        test_size=validation_split,
        random_state=42,
    )

    # Create Samplers
    train_sampler = SequentialSampler(train_indices)
    val_sampler = SequentialSampler(val_indices)

    # dataset = LibriTTSMMDatasetAcoustic("checkpoints/libri_preprocessed_data.pt")
    train_loader = DataLoader(
        dataset,
        # 4x80Gb max 10 sec audio
        # batch_size=20, # self.train_config.batch_size,
        # 4*80Gb max ~20.4 sec audio
        batch_size=batch_size,
        # TODO: find the optimal num_workers
        num_workers=num_workers,
        sampler=train_sampler,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    val_loader = DataLoader(
        dataset,
        # 4x80Gb max 10 sec audio
        # batch_size=20, # self.train_config.batch_size,
        # 4*80Gb max ~20.4 sec audio
        batch_size=batch_size,
        # TODO: find the optimal num_workers
        num_workers=num_workers,
        sampler=val_sampler,
        persistent_workers=True,
        pin_memory=True,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )

    return train_loader, val_loader
