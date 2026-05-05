# Modal: FlashAttention-2 vs dense PyTorch on B200 → tables/benchmark_flash_attention_modal.csv

from __future__ import annotations

import csv
import gc
import io
import itertools
import math
from pathlib import Path

import modal

GPU = "B200"
OUTPUT_CSV = Path("tables/benchmark_flash_attention_modal.csv")

SEQ_LENS = tuple(2**i for i in range(7, 17))
DIMS = tuple(2**i for i in range(4, 8))

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch~=2.11.0", "einops")
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
)

app = modal.App("cs336-systems-flash-attention-benchmark")


@app.function(image=image, gpu=GPU, timeout=7200)
def run_benchmark_remote() -> str:
    import torch
    import triton.testing

    from cs336_systems.gpu_kernels.flash_attention import FlashAttention2Triton

    def pytorch_causal_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        head_dim = q.shape[-1]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        num_queries = scores.shape[-2]
        num_keys = scores.shape[-1]
        query_indices = torch.arange(num_queries, device=scores.device)[:, None]
        key_indices = torch.arange(num_keys, device=scores.device)[None, :]
        negative_infinity = torch.tensor(-1e6, device=scores.device, dtype=scores.dtype)
        scores = torch.where(query_indices >= key_indices, scores, negative_infinity)
        attention_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attention_weights, v)

    flash_attention = FlashAttention2Triton.apply

    def bench_median_ms(benchmark_function):
        return float(triton.testing.do_bench(benchmark_function, warmup=25, rep=100, return_mode="median"))

    def safe_bench_ms(benchmark_function):
        try:
            return f"{bench_median_ms(benchmark_function):.4f}"
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                gc.collect()
                torch.cuda.empty_cache()
                return ""
            raise

    column_names = (
        "seq_len",
        "d_model",
        "dtype",
        "pytorch_fwd_ms",
        "pytorch_bwd_ms",
        "pytorch_e2e_ms",
        "triton_fwd_ms",
        "triton_bwd_ms",
        "triton_e2e_ms",
    )
    output_buffer = io.StringIO()
    csv_writer = csv.DictWriter(output_buffer, fieldnames=column_names)
    csv_writer.writeheader()

    for seq_len, head_dim, dtype in itertools.product(SEQ_LENS, DIMS, (torch.bfloat16, torch.float32)):
        torch.manual_seed(seq_len + head_dim * 2654435761 + (1 if dtype == torch.float32 else 0))
        q = torch.randn(1, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
        k = torch.randn(1, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
        v = torch.randn(1, seq_len, head_dim, device="cuda", dtype=dtype, requires_grad=True)
        grad_output = torch.randn(1, seq_len, head_dim, device="cuda", dtype=dtype)

        row = {
            "seq_len": seq_len,
            "d_model": head_dim,
            "dtype": "bfloat16" if dtype == torch.bfloat16 else "float32",
        }

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        def pytorch_forward_only():
            with torch.no_grad():
                pytorch_causal_attention(q, k, v)

        row["pytorch_fwd_ms"] = safe_bench_ms(pytorch_forward_only)

        q.grad, k.grad, v.grad = None, None, None
        pytorch_output = pytorch_causal_attention(q, k, v)
        pytorch_output.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()

        def pytorch_backward_only(saved_output=pytorch_output):
            q.grad = k.grad = v.grad = None
            saved_output.backward(grad_output, retain_graph=True)

        row["pytorch_bwd_ms"] = safe_bench_ms(pytorch_backward_only)

        del pytorch_output
        q.grad = k.grad = v.grad = None
        gc.collect()
        torch.cuda.empty_cache()

        def pytorch_end_to_end():
            q.grad = k.grad = v.grad = None
            pytorch_causal_attention(q, k, v).backward(grad_output)

        row["pytorch_e2e_ms"] = safe_bench_ms(pytorch_end_to_end)

        q.grad = k.grad = v.grad = None
        gc.collect()
        torch.cuda.empty_cache()

        def triton_forward_only():
            with torch.no_grad():
                flash_attention(q, k, v, True)

        row["triton_fwd_ms"] = safe_bench_ms(triton_forward_only)

        q.grad = k.grad = v.grad = None
        triton_output = flash_attention(q, k, v, True)
        triton_output.backward(grad_output, retain_graph=True)
        torch.cuda.synchronize()

        def triton_backward_only(saved_output=triton_output):
            q.grad = k.grad = v.grad = None
            saved_output.backward(grad_output, retain_graph=True)

        row["triton_bwd_ms"] = safe_bench_ms(triton_backward_only)

        del triton_output
        q.grad = k.grad = v.grad = None
        gc.collect()
        torch.cuda.empty_cache()

        def triton_end_to_end():
            q.grad = k.grad = v.grad = None
            flash_attention(q, k, v, True).backward(grad_output)

        row["triton_e2e_ms"] = safe_bench_ms(triton_end_to_end)

        csv_writer.writerow(row)

    return output_buffer.getvalue()


@app.local_entrypoint()
def main() -> None:
    csv_body = run_benchmark_remote.remote()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.write_text(csv_body)
    print(csv_body)
