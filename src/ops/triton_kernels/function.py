import torch
import triton
from typing import Any
from torch.autograd import Function
from torch.cuda.amp.autocast_mode import custom_bwd, custom_fwd
from .forward import forward_kernel
from .backward import backward_kernel



class DCNFunction(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx: Any, inputs, deformables, weights) -> Any:
        B, H, W, G, C = inputs.shape
        _, _, _, _, K, _ = deformables.shape
        out = torch.zeros_like(inputs)
        grid = lambda META: (B * H * W * G,)

        forward_kernel[grid](B, H, W, G, C, K, inputs, deformables, weights, out)
        ctx.save_for_backward(inputs, deformables, weights)
        return out

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = grad_outputs[0].contiguous()

        inputs, deformables, weights = ctx.saved_tensors
        B, H, W, G, C = inputs.shape
        _, _, _, _, K, _ = deformables.shape

        grad_inputs = torch.zeros_like(inputs)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        grid = lambda META: (B * H * W * G,)
        backward_kernel[grid](
            B, H, W, G, C, K,
            inputs,
            deformables,
            weights,
            grad_output,
            grad_inputs,
            grad_deformables,
            grad_weights,
        )
        return (grad_inputs, grad_deformables, grad_weights)