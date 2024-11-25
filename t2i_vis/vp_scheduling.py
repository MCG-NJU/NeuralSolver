# Copyright 2024 Stanford University Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DISCLAIMER: This code is strongly influenced by https://github.com/pesser/pytorch_diffusion
# and https://github.com/hojonathanho/diffusion

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from torch import Tensor

class VPScheduler:
    def __init__(
            self,
            beta_min=0.1,
            beta_max=20,
    ):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_max - beta_min
    def beta(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        return (self.beta_min + (self.beta_d * t)).view(-1, 1, 1, 1)

    def sigma(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        inter_beta:Tensor = 0.5*self.beta_d*t**2 + self.beta_min* t
        return (1-torch.exp_(-inter_beta)).sqrt().view(-1, 1, 1, 1)

    def alpha(self, t) -> Tensor:
        t = torch.clamp(t, min=1e-3, max=1)
        inter_beta: Tensor = 0.5 * self.beta_d * t ** 2 + self.beta_min * t
        return torch.exp(-0.5*inter_beta).view(-1, 1, 1, 1)

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

@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->DDIM
class SchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`torch.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None



class NeuralSolver(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.vp_scheduler = VPScheduler()
        self.init_noise_sigma = 1.0
        self.buffer = []
    def scale_model_input(self, latents, t):
        return latents
    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self._index = 0
        self._timedeltas = torch.tensor(DDPM_DATA[f"step{num_inference_steps}"]["timedeltas"], dtype=torch.float32, device=device)
        self._coeffs = torch.tensor(DDPM_DATA[f"step{num_inference_steps}"]["coeffs"], dtype=torch.float32, device=device)
        self._contiguous_timestep = [0.999,]
        for t in range(num_inference_steps-1):
            self._contiguous_timestep.append(max(self._contiguous_timestep[-1] - self._timedeltas[t], 0.0))
        self.timesteps = torch.tensor(self._contiguous_timestep, dtype=torch.float32, device=device)*self.num_train_timesteps
        self.timesteps = self.timesteps.to(torch.int64)
        self._contiguous_timestep = torch.tensor(self._contiguous_timestep, dtype=torch.float32, device=device)
        self.num_inference_steps = num_inference_steps

    def step(
        self,
        eps: torch.Tensor,
        timestep: int,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[SchedulerOutput, Tuple]:
        if timestep == self.num_train_timesteps -1:
            self.buffer.clear()
            self._index = 0
        dtype = x.dtype
        t_cur = self._contiguous_timestep[self._index]
        dt = self._timedeltas[self._index]
        sigma = self.vp_scheduler.sigma(t_cur)
        alpha = self.vp_scheduler.alpha(t_cur)
        lamda = (alpha / sigma)
        sigma_next = self.vp_scheduler.sigma(t_cur - dt)
        alpha_next = self.vp_scheduler.alpha(t_cur - dt)
        lamda_next = (alpha_next / sigma_next)
        x0 = (x - sigma * eps) / alpha
        self.buffer.append(x0)
        dpmx = torch.zeros_like(x0)
        sum_solver_coeff = 0.0
        for j in range(self._index):
             dpmx += self._coeffs[self._index, j] * self.buffer[j]
             sum_solver_coeff += self._coeffs[self._index, j]
        dpmx += (1 - sum_solver_coeff) * self.buffer[-1]
        delta_lamda = lamda_next - lamda
        x = (sigma_next / sigma) * x + sigma_next * (delta_lamda) * dpmx
        x = x.to(dtype)
        self._index += 1
        if not return_dict:
            return (x,)
        return SchedulerOutput(prev_sample=x, pred_original_sample=x0)


    def __len__(self):
        return self.num_train_timesteps
