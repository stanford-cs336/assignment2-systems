# Modal: naive per-parameter all-reduce DDP — XL, 2 GPUs.

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

app = modal.App("cs336-systems-naive-ddp-benchmark")

OUTPUT_CSV = Path("tables/naive_ddp_benchmark_modal.csv")


@app.function(image=image, gpu=GPU, timeout=3600)
def run_benchmark_remote() -> str:
    import torch
    import torch.multiprocessing as torch_mp

    from cs336_systems.distributed_training.ddp_xl_bench_core import naive_ddp_xl_bench_rank

    try:
        py_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    context = py_mp.get_context("spawn")
    manager = context.Manager()
    result_queue = manager.Queue()
    port = 29633

    torch_mp.spawn(naive_ddp_xl_bench_rank, args=(2, port, result_queue), nprocs=2, join=True)
    mean_total_ms, mean_comm_ms, comm_fraction = result_queue.get()

    output_buffer = io.StringIO()
    csv_writer = csv.writer(output_buffer)
    csv_writer.writerow(["setting", "mean_step_ms_rank_mean", "mean_grad_comm_ms_rank_mean", "frac_time_comm"])
    csv_writer.writerow(
        ["xl_ctx2048_bf16_naive_ddp_2gpu_bs2", f"{mean_total_ms:.4f}", f"{mean_comm_ms:.4f}", f"{comm_fraction:.4f}"]
    )
    return output_buffer.getvalue()


@app.local_entrypoint()
def main() -> None:
    csv_text = run_benchmark_remote.remote()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(csv_text)
    print(csv_text)
