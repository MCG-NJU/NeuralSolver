import torch
import torch.nn as nn
import numpy as np
import math

from torch.nn.init import zeros_
from src.models.base_model import BaseModel
from src.ops.triton_kernels.function import DCNFunction

def modulate(x, shift, scale):
    return x * (1 + scale[:, None, None]) + shift[:, None, None]

class PatchEmbed(nn.Module):
    def __init__(
            self,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
    def forward(self, x):
        b, h, w, c = x.shape
        x = x.view(b, h*w, c)
        x =  self.w2(torch.nn.functional.silu(self.w1(x)) * self.w3(x))
        x = x.view(b, h, w, c)
        return x


class MultiScaleDCN(nn.Module):
    def __init__(self, in_channels, groups, channels, kernels, deformable_biass=True):
        super().__init__()
        self.in_channels = in_channels
        self.groups = groups
        self.channels = channels
        self.kernels = kernels
        self.v = nn.Linear(in_channels, groups * channels, bias=True)
        self.qk_deformables = nn.Linear(in_channels, groups * kernels * 2, bias=True)
        self.qk_scales = nn.Linear(in_channels, groups * kernels, bias=False)
        self.qk_weights = nn.Linear(in_channels, groups*kernels, bias=True)
        self.out = nn.Linear(groups * channels, in_channels)
        self.deformables_prior = nn.Parameter(torch.randn((1, 1, 1, 1, kernels, 2)), requires_grad=False)
        self.deformables_scale = nn.Parameter(torch.ones((1, 1, 1, groups, 1, 1)), requires_grad=True)
        self.max_scale = 6
        self._init_weights()
    def _init_weights(self):
        zeros_(self.qk_deformables.weight.data)
        zeros_(self.qk_scales.weight.data)
        zeros_(self.qk_deformables.bias.data)
        zeros_(self.qk_weights.weight.data)
        zeros_(self.v.bias.data)
        zeros_(self.out.bias.data)
        num_prior = int(self.kernels ** 0.5)
        dx = torch.linspace(-1, 1, num_prior, device="cuda")
        dy = torch.linspace(-1, 1, num_prior, device="cuda")
        dxy = torch.meshgrid([dx, dy], indexing="xy")
        dxy = torch.stack(dxy, dim=-1)
        dxy = dxy.view(-1, 2)
        self.deformables_prior.data[..., :num_prior*num_prior, :] = dxy
        for i in range(self.groups):
           scale = (i+1)/self.groups - 0.0001
           inv_scale = math.log((scale)/(1-scale))
           self.deformables_scale.data[..., i, :, :] = inv_scale
    def forward(self, x):
        B, H, W, _ = x.shape
        v = self.v(x).view(B, H, W, self.groups, self.channels)
        deformables = self.qk_deformables(x).view(B, H, W, self.groups, self.kernels, 2)
        scale = self.qk_scales(x).view(B, H, W, self.groups, self.kernels, 1) + self.deformables_scale
        deformables = (deformables + self.deformables_prior ) * scale.sigmoid()*self.max_scale
        weights = self.qk_weights(x).view(B, H, W, self.groups, self.kernels)
        out = DCNFunction.apply(v, deformables, weights)
        out = out.view(B, H, W, -1)
        out = self.out(out)
        return out

class FlowDCNBlock(nn.Module):
    def __init__(self, hidden_size, groups, kernels=9, mlp_ratio=4.0, deformable_biass=True):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MultiScaleDCN(hidden_size, groups=groups, channels=hidden_size//groups, kernels=kernels, deformable_biass=deformable_biass)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa[:, None, None] * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp[:, None, None] * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x




class FlowDCN(BaseModel):
    def __init__(self, deformable_biass=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.blocks = nn.ModuleList([
            FlowDCNBlock(self.hidden_size, self.num_groups, kernels=9, deformable_biass=deformable_biass) for _ in range(self.num_blocks)
        ])
        self.x_embedder = PatchEmbed(self.patch_size, self.in_channels, self.hidden_size, bias=True)
        self.initialize_weights()

    def unpatchify(self, x, h, w):
        c = self.out_channels
        p = self.x_embedder.patch_size
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        batch_size = x.shape[0]
        x = self.x_embedder(x)  # (N, D, h, w)
        height, width = x.shape[2:]
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height*width, -1)
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        B, L, C = x.shape
        x = x.view(B, height, width, C)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = x.view(B, L, C)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, height, width)  # (N, out_channels, H, W)
        if self.learn_sigma:
            x, _ = torch.split(x, self.out_channels // 2, dim=1)
        return x