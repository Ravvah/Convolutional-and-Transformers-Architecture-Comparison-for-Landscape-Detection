from typing import Dict
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from loguru import logger

from torchvision.datasets import ImageFolder
from neural_cv.data.processor import Processor
import os


class DataBuilder:

    def __init__(self, data_dir: str, batch_size: int = 32, train_size: float = 0.8, val_size: float = 0.1, test_size: float = 0.1, seed: int = 42):
        if not(0 < train_size < 1):
            raise ValueError("train_size must be between 0 and 1")

        if abs(train_size + val_size + test_size - 1.0) > 1e-6:
            raise ValueError("train + val + test must sum to 1")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seed = seed

    
    def build_dataset(self) -> Dict[str, Dict[str, DataLoader]]:
        torch.manual_seed(self.seed)

        base_dataset = ImageFolder(self.data_dir)

        indices = torch.randperm(len(base_dataset))
        n = len(indices)

        n_train = int(self.train_size * n)
        n_val = int((1 - self.train_size - 0.1) * n)

        train_idx = indices[: n_train]
        val_idx = indices[n_train: n_train + n_val]
        test_idx = indices[n_train + n_val:]

        texture_dataset = ImageFolder(self.data_dir, transform=Processor(mode="texture"))
        global_dataset = ImageFolder(self.data_dir, transform=Processor(mode="global"))

        dataset_map = {
            "texture": self._create_loaders(texture_dataset, train_idx, val_idx, test_idx),
            "global": self._create_loaders(global_dataset, train_idx, val_idx, test_idx)
        }
        logger.info("DataLoaders texture and global created !")

        return dataset_map

    
    def _create_loaders(self, dataset: ImageFolder, train_idx: Tensor, val_idx: Tensor, test_idx: Tensor) -> Dict[str, DataLoader]:
        train_dataset = Subset(dataset=dataset, indices=train_idx)
        val_dataset = Subset(dataset=dataset, indices=val_idx)
        test_dataset = Subset(dataset=dataset, indices=test_idx)

        data_loaders_map = {
            "train": DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers= os.cpu_count(), persistent_workers=True),
            "validation": DataLoader(dataset=val_dataset, batch_size=self.batch_size, num_workers= os.cpu_count(), persistent_workers=True),
            "test": DataLoader(dataset=test_dataset, batch_size=self.batch_size, num_workers= os.cpu_count(), persistent_workers=True)
        }
        return data_loaders_map



