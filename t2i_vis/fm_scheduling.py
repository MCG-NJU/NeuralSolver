# Copyright 2024 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput, logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


FM_DATA_DCN = dict(
    step5=dict(
        timedeltas=[0.0521, 0.1475, 0.2114, 0.2797, 0.3092],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.2600,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.3800, -2.2600,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.9200,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.7000,  0.0000]]
    ),
    step6=dict(
        timedeltas=[0.0391, 0.0924, 0.1650, 0.2015, 0.2511, 0.2511],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.2200,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.1200, -2.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.3000,  0.9000, -1.5600,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.7400,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -0.6200,  0.0000]]
    ),
    step7=dict(
        timedeltas=[0.0387, 0.0748, 0.1030, 0.1537, 0.1840, 0.2340, 0.2117],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.1100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0300, -1.9900,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0700,  0.4300, -1.5700,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.2100, -0.1500,  1.5300, -2.2900,  0.0000,  0.0000,  0.0000],
        [-0.0500,  0.0700, -0.2300,  0.6100, -1.3300,  0.0000,  0.0000],
        [-0.1700,  0.3100, -0.4100,  0.1700,  0.5900, -1.3100,  0.0000]]
    ),
    step8=dict(
        timedeltas=[0.0071, 0.0613, 0.0780, 0.1163, 0.1421, 0.1880, 0.2077, 0.1996],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-2.4300,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.6100, -1.5500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.9900, -0.1100, -2.0700,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0500, -0.4900,  1.3300, -1.9300,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0500, -0.3300,  0.2300,  0.7300, -1.7100,  0.0000,  0.0000,  0.0000],
        [-0.0900,  0.2500, -0.2900,  0.0500,  0.6100, -1.4500,  0.0000,  0.0000],
        [-0.2300,  0.2100, -0.0100, -0.2500,  0.2500,  0.4100, -1.2500,  0.0000]]
    ),
    step9=dict(
        timedeltas=[0.0017, 0.0510, 0.0636, 0.0911, 0.1007, 0.1443, 0.1694, 0.1910, 0.1872],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-6.1900,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.1100, -0.8100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.7300, -0.1700, -1.3700,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.3100, -0.0500,  0.1900, -1.4500,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.0300, -0.2300,  0.2900,  0.3500, -1.3500,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.1900,  0.0500,  0.0100,  0.2100,  0.2500, -1.2300,  0.0000,  0.0000,
          0.0000],
        [-0.2300,  0.2100, -0.1300,  0.1700,  0.0900,  0.0900, -1.0900,  0.0000,
          0.0000],
        [-0.1700,  0.1500,  0.1100, -0.1900,  0.0300,  0.2300,  0.1700, -1.2100,
          0.0000]]
    ),
    step10=dict(
        timedeltas=[0.0016, 0.0538, 0.0347, 0.0853, 0.0853, 0.1198, 0.1351, 0.1650, 0.1788,
        0.1406],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-7.8801,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.4000, -0.7400,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.4800, -0.1800, -0.8600,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.2600, -0.0400, -0.0400, -1.2800,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.0000, -0.0600,  0.2600,  0.2600, -1.4200,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.1000, -0.0600,  0.0800,  0.2000,  0.2200, -1.2400,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.1800,  0.1400, -0.0800,  0.1000,  0.0800,  0.1400, -1.0600,  0.0000,
          0.0000,  0.0000],
        [-0.1200,  0.1600, -0.1000,  0.0400,  0.0800,  0.0600,  0.0800, -1.0200,
          0.0000,  0.0000],
        [-0.1600,  0.0200,  0.1400,  0.0000, -0.1400,  0.0800,  0.1400,  0.3400,
         -1.3800,  0.0000]]
    )
)

FM_DATA_SIT = dict(
    step5=dict(
        timedeltas=[0.0424, 0.1225, 0.2144, 0.3073, 0.3135],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.1700,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.0700, -1.8300,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000, -0.9300,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.7100,  0.0000]]
    ),
    step6=dict(
        timedeltas=[0.0389, 0.0976, 0.1610, 0.2046, 0.2762, 0.2217],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.0400,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.6200, -2.9800,  0.0000,  0.0000,  0.0000,  0.0000],
        [-1.3200,  2.5200, -2.0400,  0.0000,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000, -0.7600,  0.0000,  0.0000],
        [ 0.0000,  0.0000,  0.0000,  0.0000, -0.6600,  0.0000]]
    ),
    step7=dict(
        timedeltas=[0.0299, 0.0735, 0.1119, 0.1451, 0.1959, 0.2698, 0.1738],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.9300,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 1.2300, -2.3100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.5900,  1.5300, -2.0900,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0900, -0.0700,  0.9900, -1.9100,  0.0000,  0.0000,  0.0000],
        [ 0.0500, -0.2100,  0.0900,  0.5500, -1.4700,  0.0000,  0.0000],
        [-0.0500,  0.1900, -0.3100,  0.3700,  0.6700, -1.7900,  0.0000]]
    ),
    step8=dict(
        timedeltas=[0.0303, 0.0702, 0.0716, 0.1112, 0.1501, 0.1833, 0.2475, 0.1358],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.9200,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.7800, -1.7000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [ 0.0600,  0.5200, -1.7600,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0200, -0.1600,  0.9800, -1.8000,  0.0000,  0.0000,  0.0000,  0.0000],
        [-0.0200, -0.1200,  0.2200,  0.2400, -1.3600,  0.0000,  0.0000,  0.0000],
        [-0.1000,  0.0600, -0.0200,  0.1800,  0.1200, -1.1000,  0.0000,  0.0000],
        [-0.1600,  0.1400, -0.0200, -0.0200,  0.3800,  0.3200, -1.7200,  0.0000]]
    ),
    step9=dict(
        timedeltas=[0.0280, 0.0624, 0.0717, 0.0894, 0.1092, 0.1307, 0.1729, 0.2198, 0.1159],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.9300,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.6300, -1.2900,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.3900, -0.1100, -1.4100,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.0700, -0.0500,  0.8300, -1.5900,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000],
        [ 0.0700, -0.1100,  0.2700,  0.2700, -1.5300,  0.0000,  0.0000,  0.0000,
          0.0000],
        [-0.0500,  0.0300,  0.0100,  0.1500,  0.1700, -1.1500,  0.0000,  0.0000,
          0.0000],
        [-0.2100,  0.2700, -0.0700, -0.0300,  0.1900,  0.0900, -0.9900,  0.0000,
          0.0000],
        [-0.1500,  0.1500,  0.0300, -0.0900,  0.2500,  0.2500,  0.2100, -1.7100,
          0.0000]]
    ),
    step10=dict(
        timedeltas=[0.0279, 0.0479, 0.0646, 0.0659, 0.1045, 0.1066, 0.1355, 0.1622, 0.1942,
        0.0908],
        coeffs=[[ 0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.9500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.5900, -1.1700,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.3500, -0.1100, -1.4500,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.1300,  0.0100,  0.7500, -1.4900,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.0500, -0.0500,  0.3100,  0.2900, -1.5900,  0.0000,  0.0000,  0.0000,
          0.0000,  0.0000],
        [ 0.0500, -0.0300, -0.0900,  0.2300,  0.1700, -1.1900,  0.0000,  0.0000,
          0.0000,  0.0000],
        [-0.0300,  0.0700, -0.0900, -0.0300,  0.2700, -0.0300, -0.9100,  0.0000,
          0.0000,  0.0000],
        [-0.1500,  0.1700,  0.0300, -0.0900,  0.0500,  0.0900,  0.0500, -0.7900,
          0.0000,  0.0000],
        [-0.1700,  0.1100,  0.1500,  0.0300,  0.0500,  0.2500,  0.0500, -0.0700,
         -1.4900,  0.0000]]
    )
)


FM_DATA = FM_DATA_SIT




class NeuralSolver(SchedulerMixin, ConfigMixin):
    """
    Euler scheduler.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        shift (`float`, defaults to 1.0):
            The shift value for the timestep schedule.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.buffer = []

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        self._index = 0
        self._timedeltas = torch.tensor(FM_DATA[f"step{num_inference_steps}"]["timedeltas"], dtype=torch.float32,
                                        device=device)
        self._coeffs = torch.tensor(FM_DATA[f"step{num_inference_steps}"]["coeffs"], dtype=torch.float32,
                                    device=device)
        self._contiguous_timestep = [1.0, ]
        for t in range(num_inference_steps - 1):
            self._contiguous_timestep.append(self._contiguous_timestep[-1] - self._timedeltas[t])
        self.timesteps = torch.tensor(self._contiguous_timestep, dtype=torch.float32, device=device)
        self.timesteps = self.timesteps*self.num_train_timesteps
        self._contiguous_timestep = torch.tensor(self._contiguous_timestep, dtype=torch.float32, device=device)
        self.num_inference_steps = num_inference_steps

    def step(
            self,
            v: torch.Tensor,
            timestep: int,
            x: torch.Tensor,
            return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        if int(timestep) == self.num_train_timesteps:
            self.buffer.clear()
            self._index = 0
        dtype = x.dtype
        dt = self._timedeltas[self._index]  # .to(x.device, x.dtype)
        self.buffer.append(v)
        v = torch.zeros_like(v)
        sum_solver_coeff = 0
        for j in range(self._index):
            v += self._coeffs[self._index, j] * self.buffer[j]
            sum_solver_coeff += self._coeffs[self._index, j]
        v += (1 - sum_solver_coeff) * self.buffer[-1]
        x = x - v * dt
        x = x.to(dtype)
        self._index += 1
        if not return_dict:
            return (x,)
        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=x)

    def __len__(self):
        return self.num_train_timesteps
