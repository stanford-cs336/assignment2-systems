import gc
from collections import defaultdict
from pathlib import Path

import modal
import torch
import torch.nn as nn

from cs336_basics.model import RotaryEmbedding, TransformerBlock
from cs336_systems.profiling_benchmarking.benchmarking_script import GPU, image

app = modal.App("cs336-systems-block-memory-attribution")

D_MODEL, D_FF, N_HEADS, CTX = 2560, 10240, 32, 2048
BATCH = 4


def measure_block_saved_tensors() -> list[tuple[str, int]]:
    block = TransformerBlock(
        d_model=D_MODEL,
        d_ff=D_FF,
        num_heads=N_HEADS,
        positional_encoder=RotaryEmbedding(dim=D_MODEL // N_HEADS, context_length=CTX),
    ).cuda().to(torch.float32)
    block = torch.compile(block, fullgraph=True)

    x = torch.randn(BATCH, CTX, D_MODEL, device="cuda", requires_grad=True)

    saves: list[tuple[str, int]] = []

    def pack_hook(t: torch.Tensor) -> torch.Tensor:
        if isinstance(t, nn.Parameter):
            return t
        name = type(t.grad_fn).__name__ if t.grad_fn is not None else f"leaf_{list(t.shape)}"
        saves.append((name, t.numel() * t.element_size()))
        return t

    def unpack_hook(t: torch.Tensor) -> torch.Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        y = block(x)
        y.sum().backward()

    return saves


@app.function(image=image, gpu=GPU, timeout=1800)
def run_remote() -> list[tuple[str, int]]:
    gc.collect()
    torch.cuda.empty_cache()
    return measure_block_saved_tensors()


def print_report(saves: list[tuple[str, int]]) -> None:
    total = sum(b for _, b in saves)
    by_op: dict[str, int] = defaultdict(int)
    for name, size in saves:
        by_op[name] += size

    ranked = sorted(by_op.items(), key=lambda kv: kv[1], reverse=True)
    bytes_per_mib = 1024 ** 2

    print(f"Total saved for backward: {total / bytes_per_mib:.2f} MiB  ({len(saves)} individual saves)")
    print()
    print(f"{'Grad-fn / leaf type':<48} {'MiB':>8}  {'%':>6}")
    print("-" * 68)
    for name, size in ranked[:15]:
        print(f"{name:<48} {size / bytes_per_mib:>8.1f}  {100 * size / total:>5.1f}%")


@app.local_entrypoint()
def main():
    saves = run_remote.remote()
    print_report(saves)
