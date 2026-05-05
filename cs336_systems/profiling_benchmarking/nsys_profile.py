import argparse
import subprocess
import sys
from pathlib import Path

import modal
import torch

from cs336_basics.optimizer import AdamW
from cs336_systems.profiling_benchmarking.benchmarking_script import (
    DTYPE,
    GPU,
    configure_fp32_precision,
    get_random_batch,
    initialize_model,
    run_step,
)


DEFAULT_OUTPUT_DIR = Path("profiles")
DEFAULT_OUTPUT_NAME = "nsys_profile"
DEFAULT_VOCAB_SIZE = 10_000
DEFAULT_BATCH_SIZE = 4
DEFAULT_CONTEXT_LENGTH = 256
DEFAULT_D_MODEL = 512
DEFAULT_NUM_LAYERS = 4
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF = 1344

profile_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "apt-get update && "
        "apt-get install -y $(apt-cache search '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -n 1) && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping")
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
)
app = modal.App("cs336-systems-nsys-profile")


def nsys_args(output_base: Path, gpu_metrics_device: int, include_backtraces: bool) -> list[str]:
    args = [
        "nsys",
        "profile",
        "--force-overwrite=true",
        "--trace=cuda,cudnn,cublas,osrt,nvtx",
        "--pytorch=functions-trace,autograd-shapes-nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--stats=true",
        f"--gpu-metrics-devices={gpu_metrics_device}",
        "--output",
        str(output_base),
    ]

    if include_backtraces:
        args.extend(["--cudabacktrace=all", "--python-backtrace=cuda"])

    return args


def make_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return AdamW(model.parameters())


def run_step_with_ranges(model: torch.nn.Module, batch: torch.Tensor, method: str, optimizer: torch.optim.Optimizer):
    optimizer.zero_grad(set_to_none=True)

    torch.cuda.nvtx.range_push("forward")
    logits = model(batch)
    torch.cuda.nvtx.range_pop()

    if method in ["fb", "fbo"]:
        torch.cuda.nvtx.range_push("backward")
        loss = logits.mean()
        loss.backward()
        torch.cuda.nvtx.range_pop()

    if method == "fbo":
        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()
        torch.cuda.nvtx.range_pop()


def run_profiled_workload(
    warmup_steps: int,
    timed_steps: int,
    method: str,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
):
    configure_fp32_precision()
    model = initialize_model(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    ).cuda().to(DTYPE)
    batch = get_random_batch(batch_size=batch_size, vocab_size=vocab_size, context_length=context_length).cuda()
    optimizer = make_optimizer(model)

    for _ in range(warmup_steps):
        run_step(model, batch, method, optimizer)

    torch.cuda.cudart().cudaProfilerStart()
    for step in range(timed_steps):
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"{method}_step_{step}")
        run_step_with_ranges(model, batch, method, optimizer)
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()


def run_nsys_subprocess(
    output_dir: Path,
    output_name: str,
    warmup_steps: int,
    timed_steps: int,
    method: str,
    gpu_metrics_device: int,
    include_backtraces: bool,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> tuple[Path, subprocess.CompletedProcess[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / output_name
    command = [
        *nsys_args(output_base, gpu_metrics_device, include_backtraces),
        "--",
        sys.executable,
        "-m",
        "cs336_systems.profiling_benchmarking.nsys_profile",
        "--worker",
        "--warmup-steps",
        str(warmup_steps),
        "--timed-steps",
        str(timed_steps),
        "--method",
        method,
        "--batch-size",
        str(batch_size),
        "--vocab-size",
        str(vocab_size),
        "--context-length",
        str(context_length),
        "--d-model",
        str(d_model),
        "--num-layers",
        str(num_layers),
        "--num-heads",
        str(num_heads),
        "--d-ff",
        str(d_ff),
    ]

    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return output_base.with_suffix(".nsys-rep"), result


@app.function(image=profile_image, gpu=GPU, timeout=3600)
def run_nsys_profile_remote(
    warmup_steps: int,
    timed_steps: int,
    method: str,
    output_name: str,
    gpu_metrics_device: int,
    include_backtraces: bool,
    batch_size: int,
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
) -> tuple[str, bytes, str, str]:
    report_path, result = run_nsys_subprocess(
        output_dir=Path("/tmp/nsys_profiles"),
        output_name=output_name,
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        method=method,
        gpu_metrics_device=gpu_metrics_device,
        include_backtraces=include_backtraces,
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    return report_path.name, report_path.read_bytes(), result.stdout, result.stderr


def main_cli():
    parser = argparse.ArgumentParser(description="Profile the CS336 transformer benchmark with NVIDIA Nsight Systems.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--timed-steps", type=int, default=20)
    parser.add_argument("--method", choices=["f", "fb", "fbo"], default="f")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--gpu-metrics-device", type=int, default=0)
    parser.add_argument("--include-backtraces", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH)
    parser.add_argument("--d-model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--d-ff", type=int, default=DEFAULT_D_FF)
    args = parser.parse_args()

    if args.worker:
        run_profiled_workload(
            warmup_steps=args.warmup_steps,
            timed_steps=args.timed_steps,
            method=args.method,
            batch_size=args.batch_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
        )
        return

    report_path, result = run_nsys_subprocess(
        output_dir=args.output_dir,
        output_name=args.output_name,
        warmup_steps=args.warmup_steps,
        timed_steps=args.timed_steps,
        method=args.method,
        gpu_metrics_device=args.gpu_metrics_device,
        include_backtraces=args.include_backtraces,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    print(f"Wrote {report_path}")


@app.local_entrypoint()
def main(
    warmup_steps: int = 5,
    timed_steps: int = 20,
    method: str = "f",
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    output_name: str = DEFAULT_OUTPUT_NAME,
    gpu_metrics_device: int = 0,
    include_backtraces: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    d_model: int = DEFAULT_D_MODEL,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    d_ff: int = DEFAULT_D_FF,
):
    report_name, report_bytes, stdout, stderr = run_nsys_profile_remote.remote(
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        method=method,
        output_name=output_name,
        gpu_metrics_device=gpu_metrics_device,
        include_backtraces=include_backtraces,
        batch_size=batch_size,
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    output_path = Path(output_dir) / report_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(report_bytes)

    print(stdout, end="")
    print(stderr, end="", file=sys.stderr)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main_cli()
