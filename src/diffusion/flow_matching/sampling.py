import torch

from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *

from typing import Callable


def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

def sde_mean_step_fn(x, v, dt, s, w):
    return x + v * dt + s * w * dt

def sde_step_fn(x, v, dt, s, w):
    return x + v*dt + s * w* dt + torch.sqrt(2*w*dt)*torch.randn_like(x)

def sde_preserve_step_fn(x, v, dt, s, w):
    return x + v*dt + 0.5*s*w* dt + torch.sqrt(w*dt)*torch.randn_like(x)


import logging
logger = logging.getLogger(__name__)

class EulerSampler(BaseSampler):
    def __init__(
            self,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            last_step=None,
            last_step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.step_fn = step_fn
        self.last_step = last_step
        self.last_step_fn = last_step_fn
        self.w_scheduler = w_scheduler

        if self.last_step is None or self.num_steps == 1:
            self.last_step = 1.0 / self.num_steps
        assert self.last_step > 0.0
        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")

    def _impl_sampling(self, net, images, labels):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = images.shape[0]
        steps = torch.linspace(0.0, 1 - self.last_step, self.num_steps, device=images.device)
        steps = torch.cat([steps, torch.tensor([1.0], device=images.device)], dim=0)

        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        dt = steps[1] - steps[0]

        trajs = [x, ]
        for i, t_cur in enumerate(steps[:-1]):
            t_cur = t_cur.repeat(batch_size)
            sigma = self.scheduler.sigma(t_cur)
            dalpha_over_alpha = self.scheduler.dalpha_over_alpha(t_cur)
            dsigma_mul_sigma = self.scheduler.dsigma_mul_sigma(t_cur)
            if self.w_scheduler:
                w = self.w_scheduler.w(t_cur)
            else:
                w = 0.0

            cfg_x = torch.cat([x, x], dim=0)
            t = t_cur.repeat(2)
            out = net(cfg_x, t, labels)
            out = self.guidance_fn(out, self.guidance)
            v = out
            s = ((1/dalpha_over_alpha)*v - x)/(sigma**2 - (1/dalpha_over_alpha)*dsigma_mul_sigma)
            if i < self.num_steps -1 :
                x = self.step_fn(x, v, dt, s=s, w=w)
            else:
                x = self.last_step_fn(x, v, self.last_step, s=s, w=w)
            trajs.append(x)
        return trajs