"""NCCL all-reduce microbenchmark worker for ``torch.multiprocessing.spawn`` (used from Modal)."""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.distributed as dist

_FP32_BYTES = 4
_WARMUP_ROUNDS = 5
_TIMED_ROUNDS = 25


def allreduce_benchmark_rank(rank: int, world_size: int, nbytes: int, port: int, result_queue) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    num_fp32 = nbytes // _FP32_BYTES
    tensor = torch.zeros(num_fp32, device=device, dtype=torch.float32)
    try:
        dist.barrier()
        for _ in range(_WARMUP_ROUNDS):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        dist.barrier()

        latency_seconds: list[float] = []
        for _ in range(_TIMED_ROUNDS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize()
            latency_seconds.append(time.perf_counter() - t0)

        local_median_s = float(statistics.median(latency_seconds))
        gathered: list[float | None] = [None] * world_size
        dist.all_gather_object(gathered, local_median_s)
        if rank == 0:
            mean_of_medians_s = statistics.mean(m for m in gathered if m is not None)
            result_queue.put((world_size, nbytes, mean_of_medians_s))
        dist.barrier()
    finally:
        dist.destroy_process_group()
