"""torchrun worker for DDP nsys profiling (naive or overlap)."""
from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.distributed_training.naive_ddp import NaiveDDP, naive_ddp_sync_gradients
from cs336_systems.distributed_training.overlapping_ddp import OverlappingDDP

XL = {
    "vocab_size": 10_000,
    "context_length": 2048,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
}

WARMUP_STEPS = 3
TIMED_STEPS = 2


def run_step(model, batch, optimizer, mode: str, profile: bool) -> None:
    optimizer.zero_grad(set_to_none=True)

    if profile:
        torch.cuda.nvtx.range_push("forward")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(batch)
        loss = logits.float().mean()
    if profile:
        torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if profile:
        torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.nvtx.range_push("grad_sync")
    if mode == "overlap":
        model.finish_gradient_synchronization()
    else:
        torch.cuda.synchronize()
        naive_ddp_sync_gradients(model.module)
    torch.cuda.synchronize()
    if profile:
        torch.cuda.nvtx.range_pop()

    optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["naive", "overlap"], default="naive")
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    global_batch = 2
    local_batch = global_batch // world_size

    torch.manual_seed(0)
    base = BasicsTransformerLM(**XL).to(device=device, dtype=torch.float32)
    model = OverlappingDDP(base) if args.mode == "overlap" else NaiveDDP(base)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def get_batch(step: int) -> torch.Tensor:
        torch.manual_seed(42 + step)
        full = torch.randint(0, XL["vocab_size"], (global_batch, XL["context_length"]), device=device)
        start = rank * local_batch
        return full[start : start + local_batch]

    dist.barrier()
    for step in range(WARMUP_STEPS):
        run_step(model, get_batch(step), optimizer, args.mode, profile=False)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for step in range(TIMED_STEPS):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"step_{step}")
        run_step(model, get_batch(WARMUP_STEPS + step), optimizer, args.mode, profile=True)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
