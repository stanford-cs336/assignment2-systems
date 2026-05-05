"""
Benchmark scaffold for FlashAttention-2 vs vanilla PyTorch attention (handout § flash_benchmarking).

Run on a single GPU (e.g. B200). Uses triton.testing.do_bench.

TODO(student):
  - Fill in forward/backward/e2e timings after your implementations pass tests.
  - Tune Q_TILE_SIZE / K_TILE_SIZE (or launch kwargs) per (seq, dim) if needed.
  - Emit the results table (CSV/LaTeX/console) your writeup requires.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import torch

# TODO(student): Import your autograd functions once implemented.
# from cs336_systems.gpu_kernels.flash_attention import FlashAttention2Triton


def pytorch_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    """Plain PyTorch attention (not FlashAttention). Handout Eq. 4–6 style."""
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    s = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        nq, nk = s.shape[-2], s.shape[-1]
        row = torch.arange(nq, device=s.device)[:, None]
        col = torch.arange(nk, device=s.device)[None, :]
        s = torch.where(row >= col, s, torch.tensor(-1e6, device=s.device, dtype=s.dtype))
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, v)


def _bench(fn: Callable[[], None], *, warmup: int = 25, rep: int = 100) -> float:
    import triton.testing

    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main() -> None:
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")
    assert device.type == "cuda"

    # Handout: batch 1, causal True; powers of 2 in [128, 65536] and [16, 128].
    batch = 1
    is_causal = True
    seq_lens = [2**i for i in range(7, 17)]  # 128 .. 65536
    dims = [2**i for i in range(4, 8)]  # 16 .. 128
    dtypes = [torch.bfloat16, torch.float32]

    # TODO(student): Instantiate your Triton FlashAttention apply fn, e.g.:
    # flash_fn = FlashAttention2Triton.apply

    _ = batch, is_causal, seq_lens, dims, dtypes  # silence until wired

    for seq, dim, dtype in itertools.product(seq_lens, dims, dtypes):
        # q = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)
        # k = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)
        # v = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)
        #
        # def pt_fwd() -> None:
        #     pytorch_attention_eager(q, k, v, is_causal=is_causal)
        #
        # TODO(student): Flash forward/backward/e2e; compare to PyTorch reference.
        # fwd_ms_triton = _bench(lambda: flash_fn(q, k, v, is_causal))
        _ = seq, dim, dtype

    print(
        "TODO(student): Implement the timed loop and emit your results table "
        "(see comments at top of flash_attention_benchmark.py)."
    )


if __name__ == "__main__":
    main()
