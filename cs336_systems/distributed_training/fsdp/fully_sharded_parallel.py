from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import Embedding, Linear
from einops import einsum

_Work = Any


class FullyShardedDataParallel(nn.Module):
    """Shard cs336_basics Linear / Embedding on dim 0; keep norms replicated.

    Forward prefetch: after layer k finishes, start async all_gather for layer k+2
    (layers 0 and 1 sync-gather). Mixed-precision payloads use compute_dtype on the
    wire; master shards stay FP32.

    Backward: register_hook on each output waits the async gather for that layer, then
    prefetches layer idx-2 (last two layers in forward order sync-gather).

    finish_gradient_synchronization: reduce_scatter_tensor (SUM, divide by world size)
    for sharded layers; async all_reduce for replicated params; wait on all handles.
    """

    _shard_dim = 0

    def __init__(self, module: nn.Module, compute_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.compute_dtype = compute_dtype
        self._process_group: dist.ProcessGroup | None = None
        self._sharded_param_names: set[str] = set()

        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        if dist.is_initialized():
            self._process_group = dist.group.WORLD
            with torch.no_grad():
                for _name, param in self.module.named_parameters():
                    dist.broadcast(param.data, src=0, group=self._process_group)

        shardable_units: list[tuple[str, nn.Module]] = []
        for name, mod in self.module.named_modules():
            if isinstance(mod, (Linear, Embedding)):
                shardable_units.append((f"{name}.weight" if name else "weight", mod))

        self._units: list[nn.Module] = [mod for _, mod in shardable_units]
        num_units = len(self._units)
        self._fwd_work: list[_Work | None] = [None] * num_units
        self._fwd_bufs: list[list[torch.Tensor] | None] = [None] * num_units
        self._bwd_work: list[_Work | None] = [None] * num_units
        self._bwd_bufs: list[list[torch.Tensor] | None] = [None] * num_units

        for unit_idx, (param_name, mod) in enumerate(shardable_units):
            self._setup_sharded_unit(mod, param_name, unit_idx)

    # --- communication helpers -------------------------------------------------

    def _communication_payload(self, shard: torch.Tensor) -> torch.Tensor:
        """Return the tensor to send over the network (optionally cast to compute_dtype)."""
        tensor = shard.detach()
        return tensor.to(self.compute_dtype) if self.compute_dtype is not None else tensor

    @staticmethod
    def _wait_for_work(work_handle: _Work | None) -> None:
        if work_handle is not None:
            work_handle.wait()

    def _reset_prefetch_state(self) -> None:
        if self._world_size <= 1:
            return
        for unit_idx in range(len(self._units)):
            self._wait_for_work(self._fwd_work[unit_idx])
            self._fwd_work[unit_idx] = None
            self._fwd_bufs[unit_idx] = None
            self._wait_for_work(self._bwd_work[unit_idx])
            self._bwd_work[unit_idx] = None
            self._bwd_bufs[unit_idx] = None

    def _sync_all_gather_and_cat(self, shard: torch.Tensor) -> torch.Tensor:
        assert self._process_group is not None
        payload = self._communication_payload(shard)
        rank_buffers = [torch.empty_like(payload) for _ in range(self._world_size)]
        dist.all_gather(rank_buffers, payload, group=self._process_group)
        return torch.cat(rank_buffers, dim=self._shard_dim)

    def _launch_prefetch(
        self,
        target_idx: int,
        work_list: list[_Work | None],
        buf_list: list[list[torch.Tensor] | None],
    ) -> None:
        if self._world_size <= 1 or target_idx >= len(self._units):
            return
        assert self._process_group is not None
        payload = self._communication_payload(self._units[target_idx].weight)
        scratch_buffers = [torch.empty_like(payload) for _ in range(self._world_size)]
        self._wait_for_work(work_list[target_idx])
        work_list[target_idx] = dist.all_gather(
            scratch_buffers, payload, group=self._process_group, async_op=True
        )
        buf_list[target_idx] = scratch_buffers

    def _finish_gather_or_sync(
        self,
        shard: torch.Tensor,
        unit_idx: int,
        *,
        force_sync: bool,
        work_list: list[_Work | None],
        buf_list: list[list[torch.Tensor] | None],
    ) -> torch.Tensor:
        """Finish async prefetch for unit_idx or fall back to a blocking gather."""
        if force_sync:
            return self._sync_all_gather_and_cat(shard)
        self._wait_for_work(work_list[unit_idx])
        work_list[unit_idx] = None
        prefetch_buffers = buf_list[unit_idx]
        buf_list[unit_idx] = None
        if prefetch_buffers is None:
            return self._sync_all_gather_and_cat(shard)
        return torch.cat(prefetch_buffers, dim=self._shard_dim)

    # --- lifecycle -------------------------------------------------------------

    def _make_leaf_from_gathered(self, gathered_tensor: torch.Tensor) -> torch.Tensor:
        return gathered_tensor.float().clone().requires_grad_(True)

    def _apply_forward_gather(self, mod: nn.Module, unit_idx: int) -> None:
        if self._world_size <= 1:
            leaf = mod.weight.clone().requires_grad_(mod.weight.requires_grad)
            mod._fsdp_prepared_full = leaf  # type: ignore[attr-defined]
            mod._fsdp_full_weight_grad_src = leaf  # type: ignore[attr-defined]
            return

        gathered = self._finish_gather_or_sync(
            mod.weight,
            unit_idx,
            force_sync=unit_idx < 2,
            work_list=self._fwd_work,
            buf_list=self._fwd_bufs,
        )
        leaf = self._make_leaf_from_gathered(gathered)
        mod._fsdp_prepared_full = leaf  # type: ignore[attr-defined]
        mod._fsdp_full_weight_grad_src = leaf  # type: ignore[attr-defined]

    def _wait_for_backward_gather(self, unit_idx: int) -> None:
        if self._world_size <= 1:
            return
        num_units = len(self._units)
        self._finish_gather_or_sync(
            self._units[unit_idx].weight,
            unit_idx,
            force_sync=unit_idx >= num_units - 2,
            work_list=self._bwd_work,
            buf_list=self._bwd_bufs,
        )

    def _make_backward_output_hook(self, unit_idx: int):
        def hook(grad_output: torch.Tensor) -> torch.Tensor:
            self._wait_for_backward_gather(unit_idx)
            if unit_idx >= 2:
                self._launch_prefetch(unit_idx - 2, self._bwd_work, self._bwd_bufs)
            return grad_output

        return hook

    def _register_backward_hook_if_needed(self, output: torch.Tensor, unit_idx: int) -> None:
        if self._world_size > 1 and torch.is_grad_enabled() and output.requires_grad:
            output.register_hook(self._make_backward_output_hook(unit_idx))

    def _run_forward_with_gathered_weight(
        self,
        mod: nn.Module,
        unit_idx: int,
        compute_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        full_weight_fp32 = mod._fsdp_prepared_full  # type: ignore[attr-defined]
        weight = full_weight_fp32.to(self.compute_dtype) if self.compute_dtype is not None else full_weight_fp32
        try:
            output = compute_fn(weight)
            self._register_backward_hook_if_needed(output, unit_idx)
            return output
        finally:
            mod._fsdp_prepared_full = None  # type: ignore[attr-defined]

    def _setup_sharded_unit(self, mod: nn.Module, param_name: str, unit_idx: int) -> None:
        weight = mod.weight
        assert isinstance(weight, nn.Parameter)
        full_weight = weight.data.detach().clone()

        if self._world_size > 1:
            assert self._process_group is not None
            if full_weight.shape[self._shard_dim] % self._world_size != 0:
                raise ValueError(
                    f"Cannot shard {param_name} of shape {tuple(full_weight.shape)} "
                    f"across {self._world_size} ranks along dim {self._shard_dim}."
                )
            rank = dist.get_rank(self._process_group)
            shard = full_weight.chunk(self._world_size, dim=self._shard_dim)[rank].clone()
        else:
            shard = full_weight

        mod.weight = nn.Parameter(shard, requires_grad=weight.requires_grad)
        self._sharded_param_names.add(param_name)

        outer_self = self
        current_unit_idx = unit_idx

        if isinstance(mod, Linear):

            def forward_linear(x: torch.Tensor, m: nn.Module = mod) -> torch.Tensor:
                return outer_self._run_forward_with_gathered_weight(
                    m,
                    current_unit_idx,
                    lambda w: einsum(x, w, "... d_in, d_out d_in -> ... d_out"),
                )

            mod.forward = forward_linear  # type: ignore[method-assign]

        else:

            def forward_embedding(ids: torch.Tensor, m: nn.Module = mod) -> torch.Tensor:
                return outer_self._run_forward_with_gathered_weight(
                    m, current_unit_idx, lambda w: w[ids]
                )

            mod.forward = forward_embedding  # type: ignore[method-assign]

        mod.register_forward_pre_hook(
            lambda _mod, _inp, i=unit_idx: self._apply_forward_gather(_mod, i)
        )
        mod.register_forward_hook(
            lambda _mod, _inp, _out, i=unit_idx: self._launch_prefetch(i + 2, self._fwd_work, self._fwd_bufs)
        )

    def forward(self, *inputs, **kwargs):
        self._reset_prefetch_state()
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self) -> None:
        if not dist.is_initialized():
            return
        group = self._process_group
        assert group is not None

        if self._world_size <= 1:
            for mod in self._units:
                grad_source = getattr(mod, "_fsdp_full_weight_grad_src", None)
                if grad_source is None or grad_source.grad is None:
                    continue
                mod.weight.grad = grad_source.grad.detach().clone()
                grad_source.grad = None
            return

        pending_shards: list[tuple[_Work | None, torch.Tensor, nn.Parameter]] = []

        for mod in self._units:
            grad_source = getattr(mod, "_fsdp_full_weight_grad_src", None)
            if grad_source is None or grad_source.grad is None:
                continue
            output_shard = torch.empty_like(mod.weight)
            full_grad = grad_source.grad.detach().clone()
            grad_source.grad = None
            work_handle = dist.reduce_scatter_tensor(
                output_shard,
                full_grad,
                op=dist.ReduceOp.SUM,
                group=group,
                async_op=True,
            )
            pending_shards.append((work_handle, output_shard, mod.weight))

        replicated_params: list[tuple[_Work | None, nn.Parameter]] = []
        for name, param in self.module.named_parameters():
            if name in self._sharded_param_names or param.grad is None:
                continue
            work_handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group, async_op=True)
            replicated_params.append((work_handle, param))

        for work_handle, output_shard, weight in pending_shards:
            self._wait_for_work(work_handle)
            output_shard.div_(self._world_size)
            weight.grad = output_shard.clone()

        for work_handle, param in replicated_params:
            self._wait_for_work(work_handle)
            param.grad.div_(self._world_size)

    def gather_full_params(self) -> dict[str, torch.Tensor]:
        full_params: dict[str, torch.Tensor] = {}
        group = self._process_group
        for name, param in self.module.named_parameters():
            if name not in self._sharded_param_names or self._world_size <= 1:
                full_params[name] = param.data.detach().clone()
                continue
            assert group is not None
            rank_shards = [torch.empty_like(param) for _ in range(self._world_size)]
            dist.all_gather(rank_shards, param.contiguous(), group=group)
            full_params[name] = torch.cat(rank_shards, dim=self._shard_dim).detach().clone()
        return full_params
