from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import torch

def pytorch_attention_eager(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
) -> torch.Tensor:
    head_dim = q.shape[-1]
    scale = 1.0 / math.sqrt(head_dim)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        num_queries, num_keys = scores.shape[-2], scores.shape[-1]
        query_positions = torch.arange(num_queries, device=scores.device)[:, None]
        key_positions = torch.arange(num_keys, device=scores.device)[None, :]
        scores = torch.where(query_positions >= key_positions, scores, torch.tensor(-1e6, device=scores.device, dtype=scores.dtype))
    attention_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attention_weights, v)


def _bench(fn: Callable[[], None], *, warmup: int = 25, rep: int = 100) -> float:
    import triton.testing

    return triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median")


def main() -> None:
    from cs336_systems.gpu_kernels.flash_attention import FlashAttention2Triton

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark script.")

    batch = 1
    is_causal = True
    seq_lens = [2**i for i in range(7, 17)]
    dims = [2**i for i in range(4, 8)]
    dtypes = [torch.bfloat16, torch.float32]

    for seq, dim, dtype in itertools.product(seq_lens, dims, dtypes):
        q = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(batch, seq, dim, device=device, dtype=dtype, requires_grad=True)

        fwd_ms_pytorch = _bench(lambda: pytorch_attention_eager(q, k, v, is_causal=is_causal))
        fwd_ms_triton = _bench(lambda: FlashAttention2Triton.apply(q, k, v, is_causal))
        print(f"seq={seq} dim={dim} dtype={dtype} | pytorch={fwd_ms_pytorch:.3f}ms triton={fwd_ms_triton:.3f}ms")


if __name__ == "__main__":
    main()
