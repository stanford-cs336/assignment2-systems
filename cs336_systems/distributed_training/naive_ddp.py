"""Minimal DDP wrapper: broadcast weights from rank 0; blocking gradient ``all_reduce`` helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in self.module.parameters():
                    dist.broadcast(parameter.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _gradients_to_sync(module: nn.Module) -> list[torch.Tensor]:
    """One entry per parameter with a grad; duplicate parameter ids (tied weights) skipped."""
    seen: set[int] = set()
    out: list[torch.Tensor] = []
    for parameter in module.parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        pid = id(parameter)
        if pid in seen:
            continue
        seen.add(pid)
        out.append(parameter.grad)
    return out


def naive_ddp_sync_gradients(module: nn.Module) -> None:
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    for gradient in _gradients_to_sync(module):
        dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
        gradient.div_(world_size)


def flat_ddp_sync_gradients(module: nn.Module) -> None:
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    gradients = _gradients_to_sync(module)
    if not gradients:
        return
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    for gradient, piece in zip(gradients, torch._utils._unflatten_dense_tensors(flat, gradients), strict=True):
        gradient.copy_(piece)
