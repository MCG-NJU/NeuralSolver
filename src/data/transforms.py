import torch
import numpy as np
from typing import Callable
import torchvision.transforms as tvtf
from PIL import Image
from io import BytesIO
import base64
import pickle

class VARNeBuDataPreTransform:
    def __call__(self, sample):
        try:
            oss_path = sample[0]
            class_id = int(sample[1])
            pil_image = Image.open(BytesIO(sample[-1])).convert("RGB")
            while pil_image.size[0] >= 512 and pil_image.size[1] >= 512:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )
            return pil_image, class_id
        except Exception as e:
            print('Failed to pre-process sample: \n', repr(e))
            return None

class NeBuDataPreTransform:
    def __call__(self, sample):
        try:
            oss_path = sample[0]
            class_id = int(sample[1])
            pil_image = Image.open(BytesIO(sample[-1])).convert("RGB")
            return pil_image, class_id
        except Exception as e:
            print('Failed to pre-process sample: \n', repr(e))
            return None


class NeBuPreComputedDataPreTransform:
    def __call__(self, sample):
        try:
            base64_data = BytesIO(sample[-1]).getvalue()
            pk_data = base64.b64decode(base64_data.decode("utf-8"))
            pk_data = pickle.loads(pk_data)
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
        except Exception as e:
            print('Failed to pre-process sample: \n', repr(e))
            return torch.randn(4,32, 32), 0

class CenterCrop:
    def __init__(self, size):
        self.size = size
    def __call__(self, image):
        def center_crop_arr(pil_image, image_size):
            """
            Center cropping implementation from ADM.
            https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
            """
            while min(*pil_image.size) >= 2 * image_size:
                pil_image = pil_image.resize(
                    tuple(x // 2 for x in pil_image.size), resample=Image.BOX
                )

            scale = image_size / min(*pil_image.size)
            pil_image = pil_image.resize(
                tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
            )

            arr = np.array(pil_image)
            crop_y = (arr.shape[0] - image_size) // 2
            crop_x = (arr.shape[1] - image_size) // 2
            return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

        return center_crop_arr(image, self.size)

def xymetacollate(batch):
    latent, label, metadata = zip(*batch)
    latent = torch.stack(latent)
    label = torch.tensor(label)
    return latent, label, metadata


class UnifiedTransforms:
    def __init__(self, size, pre_transform:Callable=None, transform=(), precomputed_data=False):
        self.pre_transform = pre_transform
        if precomputed_data:
             self.transform = tvtf.Compose(
                [
                    *transform,
                    # tvtf.RandomHorizontalFlip(0.5),
                ]
            )
        else:
            self.transform = tvtf.Compose(
                [
                    *transform,
                    CenterCrop(size),
                    tvtf.RandomHorizontalFlip(0.5),
                    tvtf.ToTensor(),
                    tvtf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
                ]
            )
    def __call__(self, data):
        if self.pre_transform is not None:
            x, y = self.pre_transform(data)
        else:
            x, y = data
        x = self.transform(x)
        return x, y