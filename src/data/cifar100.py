from typing import Callable
from torchvision.datasets import CIFAR100


class LocalDataset(CIFAR100):
    def __init__(self, root:str, data_convert:Callable):
        super(LocalDataset, self).__init__(root, True)
        self.data_convert = data_convert

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        img, label = self.data_convert(data)
        return img, label
