# Modal: NCCL all-reduce latency vs tensor size and world size → CSV + plot + LaTeX.

from __future__ import annotations

import base64
import csv
import io
import itertools
import multiprocessing as py_mp
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import modal
import torch.multiprocessing as torch_mp

from cs336_systems.distributed_training.distributed_allreduce_bench_core import allreduce_benchmark_rank
from cs336_systems.distributed_training.modal._paths import repo_root

_REPO = repo_root()
_WORKSPACE = "/workspace"

GPU = "B200:6"

MB = 1024 * 1024
SIZES_BYTES = (1 * MB, 10 * MB, 100 * MB, 1024 * MB)
WORLD_SIZES = (2, 4, 6)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "matplotlib")
    .env({"PYTHONPATH": _WORKSPACE})
    .workdir(_WORKSPACE)
    .add_local_dir(str(_REPO / "cs336_systems"), remote_path=f"{_WORKSPACE}/cs336_systems")
)

app = modal.App("cs336-systems-distributed-allreduce-benchmark")

OUTPUT_CSV = Path("tables/distributed_allreduce_modal.csv")
OUTPUT_TEX = Path("tables/distributed_allreduce_modal.tex")
OUTPUT_FIG = Path("figures/distributed_allreduce_modal.png")


@app.function(image=image, gpu=GPU, timeout=900)
def run_benchmark_remote() -> dict[str, str]:
    import torch

    try:
        py_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    benchmark_results: list[tuple[int, float, float]] = []
    context = py_mp.get_context("spawn")
    manager = context.Manager()
    result_queue = manager.Queue()

    for world_size, tensor_bytes in itertools.product(WORLD_SIZES, SIZES_BYTES):
        port = 29500 + world_size * 10 + SIZES_BYTES.index(tensor_bytes)
        torch_mp.spawn(
            allreduce_benchmark_rank,
            args=(world_size, tensor_bytes, port, result_queue),
            nprocs=world_size,
            join=True,
        )
        returned_world_size, returned_bytes, latency_seconds = result_queue.get()
        benchmark_results.append((returned_world_size, returned_bytes / MB, latency_seconds * 1000.0))

    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)
    csv_writer.writerow(["world_size", "tensor_mb", "latency_mean_of_rank_medians_ms"])
    for world_size, tensor_mb, latency_ms in benchmark_results:
        csv_writer.writerow([world_size, f"{tensor_mb:g}", f"{latency_ms:.4f}"])

    tex_lines = [
        "\\begin{tabular}{rrr}",
        "\\toprule",
        "GPUs & Size (MiB) & Latency (ms) \\\\",
        "\\midrule",
    ]
    for world_size, tensor_mb, latency_ms in benchmark_results:
        tex_lines.append(f"${world_size}$ & ${tensor_mb:g}$ & ${latency_ms:.4f}$ \\\\")
    tex_lines.extend(["\\bottomrule", "\\end{tabular}", ""])

    plt.figure(figsize=(7.0, 4.2))
    for world_size in WORLD_SIZES:
        tensor_sizes_mb = [result[1] for result in benchmark_results if result[0] == world_size]
        latencies_ms = [result[2] for result in benchmark_results if result[0] == world_size]
        plt.plot(tensor_sizes_mb, latencies_ms, marker="o", label=f"{world_size} GPUs")
    plt.xlabel("Tensor size (MiB FP32)")
    plt.ylabel("All-reduce latency (ms)")
    plt.title("NCCL all-reduce (single node, FP32 zero tensors)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")
    tick_values = [1, 10, 100, 1024]
    plt.xticks(tick_values, [str(value) for value in tick_values])
    plt.tight_layout()
    png_buffer = io.BytesIO()
    plt.savefig(png_buffer, format="png", dpi=150)
    plt.close()

    return {
        "csv": csv_buffer.getvalue(),
        "tex": "\n".join(tex_lines),
        "png_b64": base64.standard_b64encode(png_buffer.getvalue()).decode("ascii"),
    }


@app.local_entrypoint()
def main() -> None:
    output = run_benchmark_remote.remote()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(output["csv"])
    OUTPUT_TEX.write_text(output["tex"])
    OUTPUT_FIG.write_bytes(base64.standard_b64decode(output["png_b64"]))
    print(output["csv"])
