import torch
import os
from typing import Callable
from torchvision.datasets import ImageFolder

class LocalDataset(ImageFolder):
    def __init__(self, root,
            data_convert:Callable,
                 ):
        super().__init__(root)
        self.data_convert = data_convert
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        img, label = self.data_convert(data)
        return img, label
import numpy as np
from torch.utils.data import Dataset
class LocalCachedDataset(Dataset):
    def __init__(self, root,):
        super().__init__()
        self.root = root
        cache_names_file = os.path.join(root, 'cache_names.txt')
        with open(cache_names_file, 'r') as f:
            self.filenames = f.readlines()
        self.filenames = sorted(self.filenames)
        self.filenames = [x.strip() for x in self.filenames]
    def __getitem__(self, idx: int):
        filename = os.path.join(self.root,self.filenames[idx])
        pk_data = torch.load(filename)
        mean = pk_data['mean']
        logvar = pk_data['logvar']
        label = pk_data['label']
        if "fliped_mean" in pk_data.keys():
            if np.random.rand() > 0.5:
                mean = pk_data['fliped_mean']
                logvar = pk_data['fliped_logvar']
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = mean + torch.randn_like(mean) * std
        return sample, label
    def __len__(self) -> int:
        return len(self.filenames)