import time
import math
import torch
from typing import Any
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from src.ops.udcn_kernels.triton_forward import _forward_kernel
from src.ops.udcn_kernels.triton_backward import _backward_kernel

from functools import lru_cache

@lru_cache()
def static_grid(H, W, device, dtype):
    grid = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy'), dim=-1)
    return grid.to(device=device, dtype=dtype)


class DeformableAttentionFunction(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx: Any, inputs, deformables, weights) -> Any:
        B, H, W, G, C = inputs.shape
        _, N, _, K = weights.shape
        out = torch.zeros(B, N, G, C, device=inputs.device, dtype=inputs.dtype)
        grid = lambda META: (B * N * G,)
        _forward_kernel[grid](B, H, W, N, G, C, K, inputs, deformables, weights, out)
        ctx.save_for_backward(inputs, deformables, weights)
        return out
    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_out = grad_outputs[0]
        values, deformables, weights = ctx.saved_tensors
        B, H, W, G, C = values.shape
        B, N, G, K = weights.shape
        grad_values = torch.zeros_like(values, dtype=torch.float16)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        grid = lambda META: (B * N * G,)
        _backward_kernel[grid](B, H, W, N, G, C, K, values, deformables, weights, grad_out, grad_values, grad_deformables, grad_weights)
        # grad_values = torch.nan_to_num(grad_values, nan=0.0, posinf=0.0, neginf=0.0)
        # grad_deformables = torch.nan_to_num(grad_deformables, nan=0.0, posinf=0.0, neginf=0.0)
        # grad_weights = torch.nan_to_num(grad_weights, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_values.to(values.dtype), grad_deformables, grad_weights