import time
import backward

import math
import torch
from typing import Any
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_fwd, custom_bwd
from .forward import forward_kernel


class DeformableAttentionFunction(Function):
    BP_FUNCS = [
        backward.backward_p1_c2_tile16_thread128,
        backward.backward_p1_c4_tile16_thread128,
        backward.backward_p2_c2_tile16_thread128,
        backward.backward_p1_c2_tile16_thread256,
        backward.backward_p1_c4_tile16_thread256,
        backward.backward_p2_c2_tile16_thread256,
        backward.backward_p1_c2_tile16_thread384,
        backward.backward_p1_c4_tile16_thread384,
        backward.backward_p2_c2_tile16_thread384,
        backward.backward_p1_c2_tile16_thread512,
        backward.backward_p1_c4_tile16_thread512,
        backward.backward_p2_c2_tile16_thread512,
        backward.backward_p1_c2_tile16_thread768,
        backward.backward_p1_c4_tile16_thread768,
        backward.backward_p2_c2_tile16_thread768,
        backward.backward_p1_c2_tile32_thread128,
        backward.backward_p1_c2_tile32_thread256,
        backward.backward_p1_c2_tile32_thread384,
        backward.backward_p1_c2_tile32_thread512,
    ]
    BP_TABLES = dict()


    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(ctx: Any, inputs, deformables, weights) -> Any:
        B, H, W, G, C = inputs.shape
        _, _, _, _, K = weights.shape
        out = torch.zeros_like(inputs)
        grid = lambda META: (B * H * W * G,)
        forward_kernel[grid](B, H, W, G, C, K, inputs, deformables, weights, out)
        ctx.save_for_backward(inputs, deformables, weights)
        return out
    @staticmethod
    def find_bp_funcs(values, deformables, weights, grad_out):
        B, H, W, G, C = values.shape
        B, H, W, G, K = weights.shape
        hash_value = 10000 * B + 100 * H + W + 1000 * G
        if hash_value in DeformableAttentionFunction.BP_TABLES.keys():
            return DeformableAttentionFunction.BP_TABLES[hash_value]
        print("missing")
        candicate_func = None
        min_t = 999.0
        grad_values = torch.zeros_like(values)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        for func in DeformableAttentionFunction.BP_FUNCS:
            t = []
            for i in range(100):
                torch.cuda.synchronize()
                start_t = time.time()
                func(B, H, W, G, K, C, values, deformables, weights, grad_out, grad_values, grad_deformables, grad_weights)
                torch.cuda.synchronize()
                t.append(time.time() - start_t)
            t = t[-50:]
            t = sum(t) / len(t)
            if t < min_t:
                min_t = t
                DeformableAttentionFunction.BP_TABLES[hash_value] = func
                candicate_func = func
        assert candicate_func is not None
        print(candicate_func)
        return candicate_func

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_out = grad_outputs[0]
        values, deformables, weights = ctx.saved_tensors
        B, H, W, G, C = values.shape
        B, H, W, G, K = weights.shape
        func = DeformableAttentionFunction.find_bp_funcs(values, deformables, weights, grad_out)
        grad_values = torch.zeros_like(values)
        grad_deformables = torch.zeros_like(deformables)
        grad_weights = torch.zeros_like(weights)
        func(B, H, W, G, K, C, values, deformables, weights, grad_out, grad_values, grad_deformables, grad_weights)
        # grad_values = torch.nan_to_num(grad_values, nan=0.0, posinf=0.0, neginf=0.0)
        # grad_deformables = torch.nan_to_num(grad_deformables, nan=0.0, posinf=0.0, neginf=0.0)
        # grad_weights = torch.nan_to_num(grad_weights, nan=0.0, posinf=0.0, neginf=0.0)
        return grad_values, grad_deformables, grad_weights