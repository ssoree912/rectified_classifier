import math
import torch as th
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint


def checkpoint(func, inputs, params, flag):
    """Compatibility wrapper used by openaimodel_medical."""
    if flag:
        return torch_checkpoint(func, *inputs, use_reentrant=False)
    return func(*inputs)


def conv_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    if dims == 2:
        return nn.Conv2d(*args, **kwargs)
    if dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    if dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    if dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return GroupNorm32(groups, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Build sinusoidal timestep embeddings.
    timesteps: Tensor[N] or Tensor[N,1]
    return: Tensor[N, dim]
    """
    if timesteps.ndim > 1:
        timesteps = timesteps.view(-1)
    timesteps = timesteps.float()
    half = dim // 2
    freqs = th.exp(-math.log(max_period) * th.arange(0, half, dtype=th.float32, device=timesteps.device) / half)
    args = timesteps[:, None] * freqs[None]
    emb = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        emb = th.cat([emb, th.zeros_like(emb[:, :1])], dim=-1)
    return emb
