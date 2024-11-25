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

DDPM_DATA = dict(
    step5=dict(
        timedeltas=[0.2582, 0.1766, 0.1766, 0.2156, 0.1731],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.4300,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.9300, -1.5500,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.6900,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.5900,  0.0000]]
    ),
    step6=dict(
        timedeltas=[0.2483, 0.1506, 0.1476, 0.1568, 0.1733, 0.1233],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.3600,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.9000, -1.8400,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0800,  0.5000, -1.0800,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.5600,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -0.5600,  0.0000]],
    ),
    step7=dict(
        timedeltas=[0.2241, 0.1415, 0.1205, 0.1158, 0.1443, 0.1627, 0.0911],
        coeffs=[[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.3800e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 1.0800e+00, -2.0200e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-2.8000e-01,  7.8000e-01, -1.5200e+00,  0.0000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [-1.4901e-08, -1.0000e-01,  6.4000e-01, -1.5000e+00,  0.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 6.0000e-02, -6.0000e-02, -6.0000e-02,  2.6000e-01, -1.0000e+00,
          0.0000e+00,  0.0000e+00],
        [ 0.0000e+00, -1.0000e-01,  2.0000e-02,  2.0000e-01,  2.6000e-01,
         -1.1200e+00,  0.0000e+00]]
    ),
    step8=dict(
        timedeltas=[0.2033, 0.1476, 0.1094, 0.0990, 0.1116, 0.1233, 0.1310, 0.0748],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.1400,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.8000, -1.7600,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0200,  0.4800, -1.6200,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.1200,  0.0600,  0.6200, -1.4200,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0400, -0.1000,  0.1200,  0.1600, -1.0400,  0.0000,  0.0000,  0.0000],
        [ 0.0600, -0.0400, -0.0600,  0.0800, -0.0800, -0.5600,  0.0000,  0.0000],
        [-0.0200, -0.0400, -0.0400,  0.1200,  0.1400,  0.0400, -0.9000,  0.0000]]
    ),
    step9=dict(
        timedeltas=[0.1959, 0.1313, 0.1142, 0.0863, 0.0898, 0.0916, 0.1119, 0.1054, 0.0735],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-1.2800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.7800, -1.6200,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.0200,  0.4400, -1.4800,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.1000,  0.1600,  0.3600, -1.3000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.0600, -0.0400,  0.2200,  0.1200, -1.0800,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.0800, -0.1000, -0.0400,  0.2400, -0.0600, -0.8600,  0.0000,  0.0000,
          0.0000],
        [ 0.0400, -0.0400, -0.0400,  0.0000,  0.0600, -0.0800, -0.5000,  0.0000,
          0.0000],
        [-0.0400,  0.0000,  0.0000, -0.0200,  0.1400,  0.0200,  0.0000, -0.7400,
          0.0000]]
    ),
    step10=dict(
        timedeltas=[0.2174, 0.1123, 0.1037, 0.0724, 0.0681, 0.0816, 0.0938, 0.0977, 0.0849,
        0.0681],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-1.1700,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.3500, -0.9900,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.2500, -0.1100, -0.9900,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.0300,  0.0500, -0.0700, -0.8500,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.0300,  0.0300,  0.2500, -0.0900, -0.9300,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.0100, -0.0300, -0.0100,  0.2100, -0.1100, -0.6700,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.0100, -0.0300, -0.0300,  0.0700,  0.0900, -0.0300, -0.8100,  0.0000,
          0.0000,  0.0000],
        [ 0.0300, -0.0300, -0.0300, -0.0300,  0.0500,  0.0100, -0.1100, -0.2700,
          0.0000,  0.0000],
        [-0.0100, -0.0100, -0.0100, -0.0100,  0.0300,  0.0700, -0.0100, -0.0500,
         -0.5700,  0.0000]]
    ),
)

class NeuralSolverSampler(BaseSampler):
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
        self._register_parameters(num_steps)
    def _register_parameters(self, num_steps=2):
        assert num_steps in [5, 6, 7, 8, 9, 10]
        data = DDPM_DATA[f"step{num_steps}"]
        self._raw_solver_coeffs = torch.tensor(data['coeffs'])
        self._raw_timedeltas = torch.tensor(data['timedeltas'])

    def _impl_sampling(self, net, images, labels):
        batch_size = images.shape[0]
        null_labels = torch.full_like(labels, self.null_class)
        labels = torch.cat([null_labels, labels], dim=0)
        x = images
        pred_trajectory = []
        trajs = [x, ]
        t_cur = torch.ones(1).to(images.device, images.dtype)*0.999
        timedeltas = self._raw_timedeltas.to(images.device, images.dtype)
        solver_coeffs = self._raw_solver_coeffs.to(images.device, images.dtype)
        t_cur = t_cur.repeat(batch_size)
        for i in range(self.num_steps):
            dt = timedeltas[i]
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
            dpmeps = torch.zeros_like(x0)
            sum_solver_coeff = 0.0
            for j in range(i):
                dpmeps += solver_coeffs[i, j] * pred_trajectory[j]
                sum_solver_coeff += solver_coeffs[i, j]
            dpmeps += (1 - sum_solver_coeff) * pred_trajectory[-1]
            delta_lamda = lamda_next - lamda
            x = (sigma_next / sigma) * x + sigma_next * (delta_lamda) * dpmeps
            pred_trajectory.append(x0)
            delta_lamda = lamda_next - lamda
            x = (sigma_next/sigma)*x + sigma_next*(delta_lamda)*x0
            t_cur = t_cur - dt
            trajs.append(x)
        return trajs