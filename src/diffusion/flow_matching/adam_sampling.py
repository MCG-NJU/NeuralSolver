import torch

from src.diffusion.base.sampling import *
from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *

from typing import Callable

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt


import logging
logger = logging.getLogger(__name__)

class AdamLMSampler(BaseSampler):
    def __init__(
            self,
            order: int = 2,
            num_steps: int = 250,
            scheduler: BaseScheduler = None,
            w_scheduler: BaseScheduler = None,
            step_fn: Callable = ode_step_fn,
            *args,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.scheduler = scheduler
        self.num_steps = num_steps
        self.step_fn = step_fn
        self.w_scheduler = w_scheduler

        assert self.scheduler is not None
        assert self.w_scheduler is not None or self.step_fn in [ode_step_fn, ]
        if self.w_scheduler is not None:
            if self.step_fn == ode_step_fn:
                logger.warning("current sampler is ODE sampler, but w_scheduler is enabled")
        self._register_parameters(order)

    # TODO: try non-uniform timesteps ?
    def _register_parameters(self, order=2):
        self._raw_solver_coeffs = torch.nn.Parameter(torch.eye(self.num_steps) * 0, requires_grad=False)
        for i in range(1, self.num_steps):
            if i >= 1 and order>=2:
                self._raw_solver_coeffs[i, i-1] = -0.5
            if i>=2 and order>=3:
                self._raw_solver_coeffs[i, i-2:i] = torch.tensor([+5 / 12, -16 / 12])
            if i>=3 and order>=4:
                self._raw_solver_coeffs[i, i - 3:i] = torch.tensor([-9 / 24, +37 / 24, -59 / 24])
        timedeltas = (1 / self.num_steps)
        self._raw_timedeltas = torch.nn.Parameter(torch.full((self.num_steps,), fill_value=timedeltas))


    def _impl_sampling(self, net, images, labels):
        """
        sampling process of Euler sampler
        -
        """
        batch_size = images.shape[0]
        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = x0 = images
        pred_trajectory = []
        trajs = [x, ]
        t_cur = torch.zeros(1).to(images.device, images.dtype)
        timedeltas = self._raw_timedeltas
        solver_coeffs = self._raw_solver_coeffs
        t_cur = t_cur.repeat(batch_size)
        for i  in range(self.num_steps):
            cfg_x = torch.cat([x, x], dim=0)
            t = t_cur.repeat(2)
            out = net(cfg_x, t, labels)
            out = self.guidance_fn(out, self.guidance)
            pred_trajectory.append(out)
            out = torch.zeros_like(out)
            sum_solver_coeff = 0.0
            for j in range(i):
                out += solver_coeffs[i, j] * pred_trajectory[j]
                sum_solver_coeff += solver_coeffs[i, j]
            out += (1-sum_solver_coeff)*pred_trajectory[-1]
            v = out
            dt = timedeltas[i]
            x0 = self.step_fn(x, v, 1-t[0], s=0, w=0)
            x = self.step_fn(x, v, dt, s=0, w=0)
            t_cur += dt
            trajs.append(x)
        return trajs