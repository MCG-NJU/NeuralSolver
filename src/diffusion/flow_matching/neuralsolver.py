import torch

from src.diffusion.base.sampling import *
from src.diffusion.base.guidance import *
from src.diffusion.base.scheduling import *

from typing import Callable

def ode_step_fn(x, v, dt, s, w):
    return x + v * dt

import logging
logger = logging.getLogger(__name__)

class NeuralSolverSampler(BaseSampler):
    def __init__(
            self,
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
        self._register_parameters(self.num_steps)

    def _register_parameters(self, num_steps):
        self._raw_solver_coeffs = nn.Parameter(torch.eye(num_steps) * 0)
        timedeltas = (1 / self.num_steps)
        self._raw_timedeltas = nn.Parameter(torch.full((num_steps,), fill_value=timedeltas))

    def _timesteps(self):
        t = torch.softmax(self._raw_timedeltas, dim=0)
        for i in range(self.num_steps):
            if i > 0:
                t[i] += t[i - 1]
        return t

    @torch.no_grad()
    def _report_stats(self):
        timedeltas = torch.softmax(self._raw_timedeltas, dim=0)
        solver_coeffs = self._raw_solver_coeffs.detach()
        return {"timedeltas": timedeltas, "solver_coeffs": solver_coeffs}


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
        trajs = [x,]
        t_cur = torch.zeros(1).to(images.device, images.dtype)
        timedeltas = self._raw_timedeltas.to(images.device, images.dtype)
        solver_coeffs = self._raw_solver_coeffs.to(images.device, images.dtype)
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
                if self.num_steps <= 6 and i == self.num_steps - 2 and j != i - 1: continue
                if self.num_steps <= 6 and i == self.num_steps - 1 and j != i - 1: continue
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