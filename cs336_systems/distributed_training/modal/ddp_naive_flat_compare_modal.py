# Modal: naive vs flat vs overlapping XL DDP (2× B200).

from __future__ import annotations

import csv
import io
import multiprocessing as py_mp
from pathlib import Path

import modal

from cs336_systems.distributed_training.modal._paths import repo_root

_REPO = repo_root()
_WORKSPACE = "/workspace"

GPU = "B200:2"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .env({"PYTHONPATH": f"{_WORKSPACE}:{_WORKSPACE}/cs336-basics"})
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping")
    .workdir(_WORKSPACE)
    .add_local_dir(str(_REPO / "cs336_systems"), remote_path=f"{_WORKSPACE}/cs336_systems")
    .add_local_dir(str(_REPO / "cs336-basics"), remote_path=f"{_WORKSPACE}/cs336-basics")
)

app = modal.App("cs336-systems-ddp-sync-compare")

OUTPUT_CSV = Path("tables/ddp_naive_vs_flat_benchmark_modal.csv")


@app.function(image=image, gpu=GPU, timeout=3600)
def run_compare_remote() -> str:
    import torch
    import torch.multiprocessing as torch_mp

    from cs336_systems.distributed_training.ddp_xl_bench_core import (
        flat_ddp_xl_bench_rank,
        naive_ddp_xl_bench_rank,
        overlap_ddp_xl_bench_rank,
    )

    try:
        py_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    context = py_mp.get_context("spawn")
    manager = context.Manager()
    benchmark_rows: list[tuple[str, float, float, float]] = []

    for label, port, rank_fn in (
        ("xl_ctx2048_bf16_naive_ddp_2gpu_bs2", 29633, naive_ddp_xl_bench_rank),
        ("xl_ctx2048_bf16_flat_ddp_2gpu_bs2", 29634, flat_ddp_xl_bench_rank),
        ("xl_ctx2048_bf16_overlap_ddp_2gpu_bs2", 29635, overlap_ddp_xl_bench_rank),
    ):
        result_queue = manager.Queue()
        torch_mp.spawn(rank_fn, args=(2, port, result_queue), nprocs=2, join=True)
        mean_total_ms, mean_comm_ms, comm_fraction = result_queue.get()
        benchmark_rows.append((label, mean_total_ms, mean_comm_ms, comm_fraction))

    output_buffer = io.StringIO()
    csv_writer = csv.writer(output_buffer)
    csv_writer.writerow(["setting", "mean_step_ms_rank_mean", "mean_grad_comm_ms_rank_mean", "frac_time_comm"])
    for label, mean_total_ms, mean_comm_ms, comm_fraction in benchmark_rows:
        csv_writer.writerow([label, f"{mean_total_ms:.4f}", f"{mean_comm_ms:.4f}", f"{comm_fraction:.4f}"])
    return output_buffer.getvalue()


@app.local_entrypoint()
def main() -> None:
    csv_text = run_compare_remote.remote()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(csv_text)
    print(csv_text)
