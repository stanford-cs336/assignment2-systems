from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(
        self,
        params,
        optimizer_cls: type[Optimizer],
        **kwargs: Any,
    ):
        self._world_size = dist.get_world_size()
        self._rank = dist.get_rank()
        self._param_owner: dict[int, int] = {}
        self._next_assignment_index = 0

        optimizer_defaults = dict(kwargs)
        param_groups = self._normalize_param_groups(params)
        super().__init__(param_groups, optimizer_defaults)

        local_params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if self._param_owner[id(param)] == self._rank
        ]
        self._inner = optimizer_cls(local_params, **optimizer_defaults)

    def _normalize_param_groups(self, params) -> list[dict]:
        if isinstance(params, torch.Tensor):
            params = [params]
        groups = list(params)
        if not isinstance(groups[0], dict):
            groups = [{"params": groups}]
        return groups

    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        new_group = dict(param_group)
        params = new_group["params"]
        params = [params] if isinstance(params, torch.Tensor) else list(params)

        for param in params:
            self._param_owner[id(param)] = self._next_assignment_index % self._world_size
            self._next_assignment_index += 1

        new_group["params"] = params
        super().add_param_group(new_group)

        if not hasattr(self, "_inner"):
            return

        local_params = [p for p in params if self._param_owner[id(p)] == self._rank]
        if not local_params:
            return

        last_group = self.param_groups[-1]
        inner_group = {key: value for key, value in last_group.items() if key != "params"}
        inner_group["params"] = local_params
        self._inner.add_param_group(inner_group)

    def step(self, closure=None, **kwargs):  # type: ignore[override]
        loss = self._inner.step(closure=closure, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                dist.broadcast(param.data, src=self._param_owner[id(param)])
        return loss
