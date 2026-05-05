"""XL DDP timing worker for torch.multiprocessing.spawn (naive / flat / overlapping sync)."""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.distributed_training.naive_ddp import NaiveDDP, flat_ddp_sync_gradients, naive_ddp_sync_gradients
from cs336_systems.distributed_training.overlapping_ddp import OverlappingDDP

XL = {
    "vocab_size": 10_000,
    "context_length": 2048,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
}

_WARMUP_STEPS = 5
_TIMED_STEPS = 25

_SYNC_FNS: dict[str, Callable[[torch.nn.Module], None]] = {
    "naive": naive_ddp_sync_gradients,
    "flat": flat_ddp_sync_gradients,
}


def xl_ddp_bench_rank(rank: int, world_size: int, port: int, result_queue, grad_sync: str) -> None:
    use_overlap = grad_sync == "overlap"
    sync_fn = None if use_overlap else _SYNC_FNS[grad_sync]

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    global_batch_size = 2
    assert global_batch_size % world_size == 0
    local_batch_size = global_batch_size // world_size

    torch.manual_seed(0)
    base_model = BasicsTransformerLM(**XL).to(device=device, dtype=torch.float32)
    model: NaiveDDP | OverlappingDDP = OverlappingDDP(base_model) if use_overlap else NaiveDDP(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    total_step_times_ms: list[float] = []
    comm_times_ms: list[float] = []

    try:
        dist.barrier()
        for step in range(_WARMUP_STEPS + _TIMED_STEPS):
            torch.manual_seed(42 + step)
            full_batch_ids = torch.randint(
                0,
                XL["vocab_size"],
                (global_batch_size, XL["context_length"]),
                device=device,
            )
            local_start = rank * local_batch_size
            local_batch = full_batch_ids[local_start : local_start + local_batch_size]

            torch.cuda.synchronize()
            step_start_time = time.perf_counter()

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(local_batch)
                loss = logits.float().mean()
            loss.backward()

            if use_overlap:
                comm_start_time = time.perf_counter()
                model.finish_gradient_synchronization()
                torch.cuda.synchronize()
                comm_end_time = time.perf_counter()
            else:
                assert sync_fn is not None
                torch.cuda.synchronize()
                comm_start_time = time.perf_counter()
                sync_fn(model.module)
                torch.cuda.synchronize()
                comm_end_time = time.perf_counter()

            optimizer.step()
            torch.cuda.synchronize()
            step_end_time = time.perf_counter()

            if step >= _WARMUP_STEPS:
                total_step_times_ms.append((step_end_time - step_start_time) * 1000.0)
                comm_times_ms.append((comm_end_time - comm_start_time) * 1000.0)

        mean_total_ms = statistics.mean(total_step_times_ms)
        mean_comm_ms = statistics.mean(comm_times_ms)
        comm_time_fraction = mean_comm_ms / mean_total_ms if mean_total_ms > 0 else float("nan")

        gathered_results: list[tuple[float, float, float] | None] = [None] * world_size
        dist.all_gather_object(gathered_results, (mean_total_ms, mean_comm_ms, comm_time_fraction))
        if rank == 0:
            valid_results = [result for result in gathered_results if result is not None]
            result_queue.put(
                (
                    statistics.mean(result[0] for result in valid_results),
                    statistics.mean(result[1] for result in valid_results),
                    statistics.mean(result[2] for result in valid_results),
                )
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def naive_ddp_xl_bench_rank(rank: int, world_size: int, port: int, result_queue) -> None:
    xl_ddp_bench_rank(rank, world_size, port, result_queue, "naive")


def flat_ddp_xl_bench_rank(rank: int, world_size: int, port: int, result_queue) -> None:
    xl_ddp_bench_rank(rank, world_size, port, result_queue, "flat")


def overlap_ddp_xl_bench_rank(rank: int, world_size: int, port: int, result_queue) -> None:
    xl_ddp_bench_rank(rank, world_size, port, result_queue, "overlap")
