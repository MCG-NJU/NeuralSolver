import torch
import torch.nn as nn
from torch import Tensor
import logging
from typing import Callable

logger = logging.getLogger(__name__)

class BaseSampler(nn.Module):
    def __init__(self,
                 null_class,
                 guidance_fn: Callable,
                 num_steps: int = 250,
                 guidance: float = 1.0,
                 weight_path: str=None,
                 *args,
                 **kwargs
        ):
        super().__init__()
        self.null_class = null_class
        self.num_steps = num_steps
        self.guidance = guidance
        self.guidance_fn = guidance_fn
        self.weight_path = weight_path

    def _timesteps(self):
        raise NotImplementedError

    def _report_stats(self):
        return {}

    def report_stats(self):
        logger.warning("logging stats of neural sampler:")
        stats = self._report_stats()
        for k, v in stats.items():
            logger.warning(f"{k}->{v}")

    def _impl_sampling(self, net, images, labels):
        raise NotImplementedError

    def __call__(self, net, images, labels, return_x_trajectory=False):
        x_trajs = self._impl_sampling(net, images, labels)
        if return_x_trajectory:
            return x_trajs
        return x_trajs[-1]