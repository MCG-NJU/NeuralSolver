import torch
import subprocess
import lightning.pytorch as pl

import logging
logger = logging.getLogger(__name__)
def class_fn_from_str(class_str):
    class_module, from_class = class_str.rsplit(".", 1)
    class_module = __import__(class_module, fromlist=[from_class])
    return getattr(class_module, from_class)


class PixelVAE(torch.nn.Module):
    def __init__(self,):
        super().__init__()
        self.model = torch.nn.Identity()

    def encode(self, x):
        return x
    def decode(self, x):
        return x

    @staticmethod
    def from_pretrained(path):
        return PixelVAE()


class LatentVAE(PixelVAE):
    def __init__(self, precompute=False, weight_path:str=None):
        super().__init__()
        self.precompute = precompute
        self.model = None
        self.weight_path = weight_path

    @torch.no_grad()
    def encode(self, x):
        assert self.model is not None
        if self.precompute:
            return x.mul_(0.18215)
        return self.model.encode(x).latent_dist.sample().mul_(0.18215)
    @torch.no_grad()
    def decode(self, x):
        assert self.model is not None
        return self.model.decode(x.div_(0.18215)).sample

    def from_pretrained(self, path):
        vae = self
        from diffusers.models import AutoencoderKL
        setattr(vae, "model", AutoencoderKL.from_pretrained(path))
        return vae

