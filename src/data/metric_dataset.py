import pathlib

import torch
import random
from torchvision.io.image import read_image
import torchvision.transforms as tvtf
from torch.utils.data import Dataset
from src.data.transforms import CenterCrop
from PIL import Image
IMG_EXTENSIONS = (
    "*.png",
    "*.JPEG",
    "*.jpeg",
    "*.jpg"
)

def test_collate(batch):
    return torch.stack(batch)

class ImageDataset(Dataset):
    def __init__(self, root, image_size=(224, 224)):
        self.root = pathlib.Path(root)
        images =  []
        for ext in IMG_EXTENSIONS:
            images.extend(self.root.rglob(ext))
        random.shuffle(images)
        self.images = list(map(lambda x: str(x), images))
        self.transform = tvtf.Compose(
            [
                CenterCrop(image_size[0]),
                tvtf.ToTensor(),
                tvtf.Lambda(lambda x: (x*255).to(torch.uint8)),
                tvtf.Lambda(lambda x: x.expand(3, -1, -1))
            ]
        )
        self.size = image_size

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx])#read_image(self.images[idx])
            image = self.transform(image)
        except Exception as e:
            print(self.images[idx])
            image = torch.zeros(3, self.size[0], self.size[1], dtype=torch.uint8)

        # print(image)
        metadata = dict(
            path = self.images[idx],
            root = self.root,
        )
        return image #, metadata

    def __len__(self):
        return len(self.images)