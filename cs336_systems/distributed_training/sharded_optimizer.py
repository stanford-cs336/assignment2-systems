from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    """Shard optimizer state: each rank updates an ~equal subset of parameters, then ``broadcast``s weights."""

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

        defaults = dict(kwargs)

        if isinstance(params, torch.Tensor):
            params = [params]
        param_groups = list(params)
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]

        super().__init__(param_groups, defaults)

        local_params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if self._param_owner[id(param)] == self._rank
        ]
        self._inner = optimizer_cls(local_params, **defaults)

    @property
    def inner_optimizer(self) -> Optimizer:
        return self._inner

    def add_param_group(self, param_group: dict[str, Any]) -> None:
        new_param_group = dict(param_group)
        params_list = new_param_group["params"]
        params_list = [params_list] if isinstance(params_list, torch.Tensor) else list(params_list)
        for param in params_list:
            self._param_owner[id(param)] = self._next_assignment_index % self._world_size
            self._next_assignment_index += 1
        new_param_group["params"] = params_list
        super().add_param_group(new_param_group)
        if hasattr(self, "_inner"):
            new_local_params = [param for param in params_list if self._param_owner[id(param)] == self._rank]
            if new_local_params:
                template_group = self.param_groups[-1]
                inner_param_group = {key: value for key, value in template_group.items() if key != "params"}
                inner_param_group["params"] = new_local_params
                self._inner.add_param_group(inner_param_group)

    def step(self, closure=None, **kwargs):  # type: ignore[override]
        loss = self._inner.step(closure=closure, **kwargs)
        for group in self.param_groups:
            for param in group["params"]:
                dist.broadcast(param.data, src=self._param_owner[id(param)])
        return loss
