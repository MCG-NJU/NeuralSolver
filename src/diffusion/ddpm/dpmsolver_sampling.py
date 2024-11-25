import torch

from src.diffusion.base.scheduling import *
from src.diffusion.base.sampling import *
from src.diffusion.base.guidance import simple_guidance_fn

from typing import Callable


def ode_step_fn(x, v, s, beta, dt):
    return x + v*dt

def sde_step_fn(x, v, s, beta, dt):
    return x + (v + 0.5*s*beta)*dt + torch.sqrt(dt*beta)*torch.randn_like(x)

import logging
logger = logging.getLogger(__name__)



class DPM1sSolverSampler(BaseSampler):
    def __init__(
            self,
            train_max_t=1000,
            num_steps: int = 250,
            scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.step_fn = step_fn
        self.train_max_t = train_max_t
        assert self.scheduler is not None


    def _impl_sampling(self, net, images, labels):
        batch_size = images.shape[0]
        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        pred_trajectory = []
        trajs = [x, ]
        t_cur = torch.ones(1).to(images.device, images.dtype)*0.999
        dt = 1/self.num_steps
        t_cur = t_cur.repeat(batch_size)
        for i in range(self.num_steps):
            sigma = self.scheduler.sigma(t_cur)
            alpha = self.scheduler.alpha(t_cur)
            lamda = (alpha/sigma)
            sigma_next = self.scheduler.sigma(t_cur - dt)
            alpha_next = self.scheduler.alpha(t_cur - dt)
            lamda_next = (alpha_next/sigma_next)
            cfg_x = torch.cat([x, x], dim=0)
            t = t_cur.repeat(2)
            eps = net(cfg_x, t * self.train_max_t, labels)
            eps = self.guidance_fn(eps, self.guidance)
            x0 = (x - sigma*eps)/alpha
            pred_trajectory.append(x0)
            delta_lamda = lamda_next - lamda
            x = (sigma_next/sigma)*x + sigma_next*(delta_lamda)*x0
            t_cur = t_cur - dt
            trajs.append(x)
        return trajs