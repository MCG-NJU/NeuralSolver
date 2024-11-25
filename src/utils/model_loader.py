from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from lightning.fabric.utilities.types import _PATH


import logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self,):
        super().__init__()

    def prepare(self, rank, world_size, device, dtype, vae, denoisers, metric, sampler):
        self._device = device
        self._dtype = dtype  # not used
        self.rank = rank
        self.world_size = world_size

        if metric.precompute_data_path:
            _precompute_data_path = dict()
            for k, v in metric.precompute_data_path.items():
                _precompute_data_path[k] = v
            metric.precompute_data_path = _precompute_data_path

    def load(self, vae, denoisers, metric, sampler):
        if vae.weight_path:
            vae = vae.from_pretrained(vae.weight_path).to(self._device)
        for i, denoiser in enumerate(denoisers):
            if denoiser.weight_path:
                weight = torch.load(denoiser.weight_path, map_location=torch.device('cpu'))
                if denoiser.load_ema and "optimizer_states" in weight.keys():
                    weight = weight["optimizer_states"][0]["ema"]
                    params = list(denoiser.parameters())
                    for w, p in zip(weight, params):
                        p.data.copy_(w)
                else:
                    try:
                        params = list(denoiser.parameters())
                        for w, p in zip(weight['state_dict'].values(), params):
                            p.data.copy_(w)
                    except:
                        denoiser.load_state_dict(weight)
            denoiser.to(self._device)
        if metric.precompute_data_path:
            metric.load_precompute_data(metric.precompute_data_path, self.rank, self.world_size)
        if sampler.weight_path:
            params = list(sampler.parameters())
            weight = torch.load(sampler.weight_path, map_location=torch.device('cpu'))
            for w, p in zip(weight['state_dict'].values(), params):
                p.data.copy_(w)
        return vae, denoisers, metric, sampler