from __future__ import annotations

import gc
import os
import tempfile
import timeit
from contextlib import nullcontext
from typing import TypedDict

import modal
import torch

from cs336_basics.model import BasicsTransformerLM

GPU = "B200"
DTYPE = torch.float32

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping")
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
)
app = modal.App("cs336-systems-benchmarking-script")


class ModelConfigDict(TypedDict):
    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int


def initialize_model(
    vocab_size: int = 10_000,
    context_length: int = 256,
    d_model: int = 512,
    num_layers: int = 4,
    num_heads: int = 8,
    d_ff: int = 1344,
) -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )


def initialize_model_from_config(config: ModelConfigDict) -> BasicsTransformerLM:
    return initialize_model(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )


def get_random_batch(batch_size: int = 4, vocab_size: int = 10_000, context_length: int = 256) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, context_length))


def configure_fp32_precision() -> None:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def run_step(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    method: str,
    optimizer: torch.optim.Optimizer,
    *,
    use_bf16_autocast: bool = False,
) -> None:
    autocast_cm = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16_autocast
        else nullcontext()
    )

    if method == "f":
        with torch.inference_mode():
            with autocast_cm:
                model(batch)
        return

    optimizer.zero_grad(set_to_none=True)
    with autocast_cm:
        logits = model(batch)
        if method in ("fb", "fbo"):
            loss = logits.mean()

    if method in ("fb", "fbo"):
        loss.backward()

    if method == "fbo":
        optimizer.step()


def capture_memory_snapshot(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    method: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    *,
    use_bf16_autocast: bool = False,
) -> tuple[bytes, int, int]:
    for _ in range(warmup_steps):
        run_step(model, batch, method, optimizer, use_bf16_autocast=use_bf16_autocast)

    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    try:
        run_step(model, batch, method, optimizer, use_bf16_autocast=use_bf16_autocast)
        torch.cuda.synchronize()
        fd, path = tempfile.mkstemp(suffix=".pickle")
        os.close(fd)
        try:
            torch.cuda.memory._dump_snapshot(path)
            with open(path, "rb") as snapshot_file:
                snapshot_bytes = snapshot_file.read()
        finally:
            os.unlink(path)
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)

    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    return snapshot_bytes, peak_allocated, peak_reserved


def benchmark_method(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    method: str,
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    timed_steps: int,
    *,
    use_bf16_autocast: bool = False,
) -> list[float]:
    for _ in range(warmup_steps):
        run_step(model, batch, method, optimizer, use_bf16_autocast=use_bf16_autocast)

    def timed_step():
        torch.cuda.synchronize()
        run_step(model, batch, method, optimizer, use_bf16_autocast=use_bf16_autocast)
        torch.cuda.synchronize()

    timer = timeit.Timer(timed_step)
    return [timer.timeit(number=1) for _ in range(timed_steps)]


@app.function(image=image, gpu=GPU, timeout=3600)
def run_benchmark_remote(
    warmup_steps: int,
    timed_steps: int,
    method: str,
    use_bf16_autocast: bool = False,
) -> list[float]:
    configure_fp32_precision()
    model = initialize_model().cuda().to(DTYPE)
    batch = get_random_batch().cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    return benchmark_method(
        model,
        batch,
        method,
        optimizer,
        warmup_steps,
        timed_steps,
        use_bf16_autocast=use_bf16_autocast,
    )


@app.local_entrypoint()
def main(
    warmup_steps: int = 5,
    timed_steps: int = 20,
    method: str = "f",
    use_bf16_autocast: bool = False,
):
    times = run_benchmark_remote.remote(warmup_steps, timed_steps, method, use_bf16_autocast)
    print(f"Method: {method}")
    print(f"use_bf16_autocast: {use_bf16_autocast}")
    print(f"Times: {times}")
