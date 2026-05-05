# Modal: XL AdamW vs ShardedOptimizer — peaks + iteration time (2× B200).

from __future__ import annotations

import csv
import io
import json
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

app = modal.App("cs336-systems-sharded-optimizer-benchmark")

OUTPUT_CSV = Path("tables/sharded_optimizer_benchmark_modal.csv")
OUTPUT_JSON = Path("tables/sharded_optimizer_benchmark_modal.json")


@app.function(image=image, gpu=GPU, timeout=3600)
def run_benchmark_remote() -> str:
    import torch.multiprocessing as torch_mp

    from cs336_systems.distributed_training.sharded_optimizer_xl_bench_core import xl_sharded_optimizer_bench_rank

    try:
        py_mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    context = py_mp.get_context("spawn")
    manager = context.Manager()
    benchmark_rows: list[dict] = []

    for use_sharded, port in ((False, 29710), (True, 29711)):
        result_queue = manager.Queue()
        torch_mp.spawn(xl_sharded_optimizer_bench_rank, args=(2, port, result_queue, use_sharded), nprocs=2, join=True)
        benchmark_rows.append(result_queue.get())

    output_fields = [
        "use_sharded",
        "peak_after_init_mb_mean",
        "peak_before_step_mb_mean",
        "peak_after_step_mb_mean",
        "mean_iter_ms_mean",
        "param_mb",
        "grad_mb_snapshot_mean",
        "opt_state_reserved_mb_mean",
    ]
    output_buffer = io.StringIO()
    csv_writer = csv.DictWriter(output_buffer, fieldnames=output_fields)
    csv_writer.writeheader()
    for row in benchmark_rows:
        csv_writer.writerow({key: row[key] for key in output_fields})
    metadata = {"runs": benchmark_rows}
    return output_buffer.getvalue() + "\n---JSON---\n" + json.dumps(metadata, indent=2)


@app.local_entrypoint()
def main() -> None:
    raw_output = run_benchmark_remote.remote()
    csv_part, _, json_part = raw_output.partition("\n---JSON---\n")
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(csv_part.strip() + "\n")
    if json_part:
        OUTPUT_JSON.write_text(json_part)
    print(csv_part)
