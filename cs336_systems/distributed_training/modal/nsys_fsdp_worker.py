"""torchrun worker for FSDP nsys profiling (XL model, 2 GPUs)."""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.distributed_training.fsdp import FullyShardedDataParallel

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


def _register_nvtx_hooks(fsdp: FullyShardedDataParallel) -> None:
    for idx, unit in enumerate(fsdp._units):
        label = f"unit_{idx}"

        def make_pre(lbl: str):
            def pre_hook(module: nn.Module, args):
                torch.cuda.nvtx.range_push(lbl)
            return pre_hook

        def make_post(lbl: str):
            def post_hook(module: nn.Module, args, output):
                torch.cuda.nvtx.range_pop()
            return post_hook

        unit.register_forward_pre_hook(make_pre(label))
        unit.register_forward_hook(make_post(label))


def run_step(fsdp: FullyShardedDataParallel, batch: torch.Tensor, optimizer: torch.optim.Optimizer, profile: bool) -> None:
    optimizer.zero_grad(set_to_none=True)

    if profile:
        torch.cuda.nvtx.range_push("forward")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = fsdp(batch)
        loss = logits.float().mean()
    if profile:
        torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if profile:
        torch.cuda.nvtx.range_pop()

    if profile:
        torch.cuda.nvtx.range_push("finish_grad_sync")
    fsdp.finish_gradient_synchronization()
    torch.cuda.synchronize()
    if profile:
        torch.cuda.nvtx.range_pop()

    optimizer.step()


def main() -> None:
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
    fsdp = FullyShardedDataParallel(base, compute_dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-4)

    _register_nvtx_hooks(fsdp)

    def get_batch(step: int) -> torch.Tensor:
        torch.manual_seed(42 + step)
        full = torch.randint(0, XL["vocab_size"], (global_batch, XL["context_length"]), device=device)
        start = rank * local_batch
        return full[start : start + local_batch]

    dist.barrier()
    for step in range(WARMUP_STEPS):
        run_step(fsdp, get_batch(step), optimizer, profile=False)
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for step in range(TIMED_STEPS):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"step_{step}")
        run_step(fsdp, get_batch(WARMUP_STEPS + step), optimizer, profile=True)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
