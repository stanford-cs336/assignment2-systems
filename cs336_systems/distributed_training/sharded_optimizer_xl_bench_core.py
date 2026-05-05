from __future__ import annotations

import os
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.distributed_training.sharded_optimizer import ShardedOptimizer

XL: dict[str, Any] = {
    "vocab_size": 10_000,
    "context_length": 2048,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
}

OPTIMIZER_KWARGS = dict(lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)


def _bytes_for_tensor(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _total_param_bytes(module: torch.nn.Module) -> int:
    return sum(_bytes_for_tensor(param) for param in module.parameters())


def _total_grad_bytes(module: torch.nn.Module) -> int:
    return sum(_bytes_for_tensor(param.grad) for param in module.parameters() if param.grad is not None)


def _total_optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total_bytes = 0
    for param_state in optimizer.state.values():
        for state_tensor in param_state.values():
            if torch.is_tensor(state_tensor):
                total_bytes += _bytes_for_tensor(state_tensor)
    return total_bytes


def xl_sharded_optimizer_bench_rank(rank: int, world_size: int, port: int, result_queue, use_sharded: bool) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    model = BasicsTransformerLM(**XL).to(device=device, dtype=torch.float32)

    torch.cuda.synchronize()
    peak_after_model_init_mb = torch.cuda.max_memory_allocated() / (1024**2)

    if use_sharded:
        optimizer = ShardedOptimizer(model.parameters(), torch.optim.AdamW, **OPTIMIZER_KWARGS)
        optimizer_for_stats = optimizer.inner_optimizer
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_KWARGS)
        optimizer_for_stats = optimizer

    global_batch_size = 2
    local_batch_size = global_batch_size // world_size
    warmup_steps = 3
    timed_steps = 15

    peak_before_optimizer_step_mb = 0.0
    peak_after_optimizer_step_mb = 0.0
    iteration_times_ms: list[float] = []

    try:
        dist.barrier()
        for step in range(warmup_steps + timed_steps):
            torch.manual_seed(42 + step)
            full_batch = torch.randint(0, XL["vocab_size"], (global_batch_size, XL["context_length"]), device=device)
            local_batch = full_batch[rank * local_batch_size : rank * local_batch_size + local_batch_size]

            if step >= warmup_steps:
                torch.cuda.synchronize()
                step_start_time = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(local_batch)
                loss = logits.float().mean()
            loss.backward()
            torch.cuda.synchronize()

            if step == warmup_steps:
                peak_before_optimizer_step_mb = torch.cuda.max_memory_allocated() / (1024**2)

            optimizer.step()
            torch.cuda.synchronize()

            if step == warmup_steps:
                peak_after_optimizer_step_mb = torch.cuda.max_memory_allocated() / (1024**2)

            if step >= warmup_steps:
                iteration_times_ms.append((time.perf_counter() - step_start_time) * 1000.0)

        param_mb = _total_param_bytes(model) / (1024**2)
        grad_mb = _total_grad_bytes(model) / (1024**2)
        optimizer_state_mb = _total_optimizer_state_bytes(optimizer_for_stats) / (1024**2)

        gathered_results = [None] * world_size
        rank_payload = (
            peak_after_model_init_mb,
            peak_before_optimizer_step_mb,
            peak_after_optimizer_step_mb,
            statistics.mean(iteration_times_ms),
            param_mb,
            grad_mb,
            optimizer_state_mb,
        )
        dist.all_gather_object(gathered_results, rank_payload)
        if rank == 0:
            valid_results = [result for result in gathered_results if result is not None]
            result_queue.put(
                {
                    "use_sharded": use_sharded,
                    "peak_after_init_mb_mean": statistics.mean(result[0] for result in valid_results),
                    "peak_before_step_mb_mean": statistics.mean(result[1] for result in valid_results),
                    "peak_after_step_mb_mean": statistics.mean(result[2] for result in valid_results),
                    "mean_iter_ms_mean": statistics.mean(result[3] for result in valid_results),
                    "param_mb": valid_results[0][4],
                    "grad_mb_snapshot_mean": statistics.mean(result[5] for result in valid_results),
                    "opt_state_reserved_mb_mean": statistics.mean(result[6] for result in valid_results),
                }
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()
