import csv
import gc
import time
from pathlib import Path

import modal
import torch
import torch.nn as nn

from cs336_basics.model import scaled_dot_product_attention

from cs336_systems.profiling_benchmarking.benchmarking_script import GPU, configure_fp32_precision, image

app = modal.App("cs336-systems-pytorch-attention-benchmark")

OUTPUT_TEX = Path("tables/pytorch_attention_benchmark.tex")
OUTPUT_CSV = Path("tables/pytorch_attention_benchmark.csv")
OUTPUT_COMPILE_TEX = Path("tables/pytorch_attention_torch_compile.tex")
OUTPUT_COMPILE_CSV = Path("tables/pytorch_attention_torch_compile.csv")

BATCH_SIZE = 8
HEAD_DIMS = (16, 32, 64, 128)
SEQ_LENS = (256, 1024, 4096, 8192, 16384)

TIME_ITERS = 100
WARMUP_FWD = 10
WARMUP_BWD = 10
COMPILE_GRAPH_WARMUP = 8


class AttentionModule(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return scaled_dot_product_attention(q, k, v)


def warmup_compiled_module(module: nn.Module, seq_len: int, head_dim: int, device: torch.device) -> None:
    for _ in range(COMPILE_GRAPH_WARMUP):
        q = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        out = module(q, k, v)
        out.sum().backward()
        torch.cuda.synchronize()


def benchmark_forward_only(attention_module: nn.Module, seq_len: int, head_dim: int, device: torch.device) -> float:
    q = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32)
    v = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32)

    for _ in range(WARMUP_FWD):
        attention_module(q, k, v)
        torch.cuda.synchronize()

    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(TIME_ITERS):
        attention_module(q, k, v)
        torch.cuda.synchronize()
    return (time.perf_counter() - start_time) / TIME_ITERS * 1000.0


def benchmark_backward_only(attention_module: nn.Module, seq_len: int, head_dim: int, device: torch.device) -> tuple[float, float]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    def make_input_tensors():
        q = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        k = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        v = torch.randn(BATCH_SIZE, seq_len, head_dim, device=device, dtype=torch.float32, requires_grad=True)
        return q, k, v

    for _ in range(WARMUP_BWD):
        q, k, v = make_input_tensors()
        out = attention_module(q, k, v)
        loss = out.sum()
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()

    q, k, v = make_input_tensors()
    out = attention_module(q, k, v)
    loss = out.sum()
    torch.cuda.synchronize()
    allocated_before_backward_bytes = torch.cuda.memory_allocated(device)

    backward_times: list[float] = []
    for _ in range(TIME_ITERS):
        q, k, v = make_input_tensors()
        out = attention_module(q, k, v)
        loss = out.sum()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward_times.append(time.perf_counter() - start_time)

    mean_backward_ms = sum(backward_times) / TIME_ITERS * 1000.0
    return mean_backward_ms, allocated_before_backward_bytes / (1024**2)


def benchmark_one_configuration(head_dim: int, seq_len: int) -> dict[str, str | float]:
    row: dict[str, str | float] = {
        "d_model": head_dim,
        "seq_len": seq_len,
        "fwd_mean_ms": "",
        "bwd_mean_ms": "",
        "mem_before_bwd_mib": "",
        "fwd_mean_ms_compiled": "",
        "bwd_mean_ms_compiled": "",
        "error": "",
        "error_compiled": "",
    }

    device = torch.device("cuda")

    vanilla_module = AttentionModule().to(device)

    torch.cuda.synchronize()
    try:
        forward_ms = benchmark_forward_only(vanilla_module, seq_len, head_dim, device)
        backward_ms, mem_before_bwd_mib = benchmark_backward_only(vanilla_module, seq_len, head_dim, device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["error"] = "OOM"
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            del vanilla_module
            gc.collect()
            torch.cuda.empty_cache()
            return row
        raise

    row["fwd_mean_ms"] = round(forward_ms, 4)
    row["bwd_mean_ms"] = round(backward_ms, 4)
    row["mem_before_bwd_mib"] = round(mem_before_bwd_mib, 2)

    del vanilla_module
    gc.collect()
    torch.cuda.empty_cache()

    compiled_module = torch.compile(AttentionModule().to(device))

    torch.cuda.synchronize()
    try:
        warmup_compiled_module(compiled_module, seq_len, head_dim, device)
        compiled_forward_ms = benchmark_forward_only(compiled_module, seq_len, head_dim, device)
        compiled_backward_ms, _mem_ignore = benchmark_backward_only(compiled_module, seq_len, head_dim, device)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            row["error_compiled"] = "OOM"
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        else:
            raise
    else:
        row["fwd_mean_ms_compiled"] = round(compiled_forward_ms, 4)
        row["bwd_mean_ms_compiled"] = round(compiled_backward_ms, 4)

    del compiled_module
    gc.collect()
    torch.cuda.empty_cache()
    return row


def run_full_benchmark_grid() -> list[dict[str, str | float]]:
    configure_fp32_precision()
    rows: list[dict[str, str | float]] = []
    for seq_len in SEQ_LENS:
        for head_dim in HEAD_DIMS:
            rows.append(benchmark_one_configuration(head_dim, seq_len))
    return rows


def write_csv(rows: list[dict[str, str | float]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    vanilla_fields = ["seq_len", "d_model", "fwd_mean_ms", "bwd_mean_ms", "mem_before_bwd_mib", "error"]
    with OUTPUT_CSV.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=vanilla_fields)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow({key: row.get(key, "") for key in vanilla_fields})

    compile_fields = [
        "seq_len",
        "d_model",
        "fwd_vanilla_ms",
        "fwd_compiled_ms",
        "bwd_vanilla_ms",
        "bwd_compiled_ms",
        "error",
        "error_compiled",
    ]
    OUTPUT_COMPILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_COMPILE_CSV.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=compile_fields)
        csv_writer.writeheader()
        for row in rows:
            csv_writer.writerow(
                {
                    "seq_len": row["seq_len"],
                    "d_model": row["d_model"],
                    "fwd_vanilla_ms": row.get("fwd_mean_ms", ""),
                    "fwd_compiled_ms": row.get("fwd_mean_ms_compiled", ""),
                    "bwd_vanilla_ms": row.get("bwd_mean_ms", ""),
                    "bwd_compiled_ms": row.get("bwd_mean_ms_compiled", ""),
                    "error": row.get("error", ""),
                    "error_compiled": row.get("error_compiled", ""),
                }
            )


def write_tex(rows: list[dict[str, str | float]]) -> None:
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{rrrrr}",
        r"  \toprule",
        r"  $T$ & $d$ & Fwd.\ mean (ms) & Bwd.\ mean (ms) & CUDA alloc.\ before bwd (MiB) \\",
        r"  \midrule",
    ]
    for row in rows:
        seq_len = int(row["seq_len"])
        head_dim = int(row["d_model"])
        vanilla_error = str(row.get("error", "")).strip()
        if vanilla_error:
            lines.append(rf"  ${seq_len}$ & ${head_dim}$ & --- & --- & --- \\")
            continue
        forward_ms = float(row["fwd_mean_ms"])
        backward_ms = float(row["bwd_mean_ms"])
        mem_mib = float(row["mem_before_bwd_mib"])
        lines.append(rf"  ${seq_len}$ & ${head_dim}$ & ${forward_ms:.4f}$ & ${backward_ms:.4f}$ & ${mem_mib:.2f}$ \\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compile_tex(rows: list[dict[str, str | float]]) -> None:
    OUTPUT_COMPILE_TEX.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"  \toprule",
        r"  $T$ & $d$ & Fwd.\ vanilla (ms) & Fwd.\ compiled (ms) & Bwd.\ vanilla (ms) & Bwd.\ compiled (ms) \\",
        r"  \midrule",
    ]
    for row in rows:
        seq_len = int(row["seq_len"])
        head_dim = int(row["d_model"])
        vanilla_error = str(row.get("error", "")).strip()
        compiled_error = str(row.get("error_compiled", "")).strip()
        if vanilla_error:
            lines.append(rf"  ${seq_len}$ & ${head_dim}$ & --- & --- & --- & --- \\")
            continue
        vanilla_forward_ms = float(row["fwd_mean_ms"])
        vanilla_backward_ms = float(row["bwd_mean_ms"])
        if compiled_error:
            lines.append(rf"  ${seq_len}$ & ${head_dim}$ & ${vanilla_forward_ms:.4f}$ & --- & ${vanilla_backward_ms:.4f}$ & --- \\")
            continue
        compiled_forward_ms = float(row["fwd_mean_ms_compiled"])
        compiled_backward_ms = float(row["bwd_mean_ms_compiled"])
        lines.append(
            rf"  ${seq_len}$ & ${head_dim}$ & ${vanilla_forward_ms:.4f}$ & ${compiled_forward_ms:.4f}$ "
            rf"& ${vanilla_backward_ms:.4f}$ & ${compiled_backward_ms:.4f}$ \\"
        )
    lines += [r"  \bottomrule", r"\end{tabular}"]
    OUTPUT_COMPILE_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


@app.function(image=image, gpu=GPU, timeout=7200, retries=0)
def run_remote() -> list[dict[str, str | float]]:
    return run_full_benchmark_grid()


@app.local_entrypoint()
def main():
    rows = run_remote.remote()
    write_csv(rows)
    write_tex(rows)
    write_compile_tex(rows)
    print(f"Wrote {OUTPUT_TEX}, {OUTPUT_COMPILE_TEX}, CSVs")
    for row in rows:
        print(row)


def argparse_main() -> None:
    rows = run_full_benchmark_grid()
    write_csv(rows)
    write_tex(rows)
    write_compile_tex(rows)
    print(f"Wrote {OUTPUT_TEX}, {OUTPUT_COMPILE_TEX}, CSVs")
    for row in rows:
        print(row)


if __name__ == "__main__":
    argparse_main()
