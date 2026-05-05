from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class OverlappingDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._pending_all_reduces: list[tuple[Any, torch.Tensor]] = []
        if dist.is_initialized():
            with torch.no_grad():
                for parameter in self.module.parameters():
                    dist.broadcast(parameter.data, src=0)
            for parameter in self.module.parameters():
                if not parameter.requires_grad:
                    continue

                def schedule_all_reduce(_unused_grad: torch.Tensor, p: nn.Parameter = parameter) -> None:
                    gradient = p.grad
                    if gradient is None:
                        return
                    work = dist.all_reduce(gradient, op=dist.ReduceOp.SUM, async_op=True)
                    self._pending_all_reduces.append((work, gradient))

                parameter.register_post_accumulate_grad_hook(schedule_all_reduce)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        for work, gradient in self._pending_all_reduces:
            work.wait()
            gradient.div_(world_size)
        self._pending_all_reduces.clear()
