# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""§8.2 — data-parallel FFN layer: backward FLOPs, comm time, bottleneck inequality.

Usage::

    uv run python cs336_systems/parallelism/problem_82_data_parallel.py
"""

from __future__ import annotations

from cs336_systems.parallelism.ring_collectives import fp16_tensor_bytes, ring_all_reduce_seconds


def backward_flops(*, batch_size: float, d_model: float, d_ff: float, num_data_parallel_ranks: int) -> float:
    """Ignore non-matmul ops; local batch is batch_size / num_data_parallel_ranks."""
    local_batch = batch_size / num_data_parallel_ranks
    return 12.0 * local_batch * d_model * d_ff


def backward_comm_seconds(
    *, d_model: float, d_ff: float, num_data_parallel_ranks: float, bandwidth_b_per_s: float
) -> float:
    """Three independent ring all-reduces on FP16 grad tensors of shape (d_model, d_ff) or (d_ff, d_model)."""
    elements_per_weight = d_model * d_ff
    bytes_per_weight = fp16_tensor_bytes(elements_per_weight)
    time_one_allreduce = ring_all_reduce_seconds(
        n_ranks=int(num_data_parallel_ranks),
        payload_bytes=bytes_per_weight,
        bandwidth_b_per_s=bandwidth_b_per_s,
    )
    return 3.0 * time_one_allreduce


def max_compute_bound_num_ranks(*, batch_size: float, compute_flops_per_s: float, bandwidth_b_per_s: float) -> float:
    """Largest N_DP where communication time is still less than or equal to compute time (ring AR)."""
    return 1.0 + batch_size * bandwidth_b_per_s / compute_flops_per_s


def main() -> None:
    print("§8.2 data_parallel_calcs")
    print("(a) Backward FLOPs (matmul only):  12 * (B/N_DP) * D * D_FF")
    print(
        "Justification: six backward matmuls each costs 2*(B/N_DP)*D*D_FF FLOPs "
        "(three dW outer products, dz = dy W3^T, and dx = dx1 W1^T + dx2 W2^T)."
    )
    print(
        "(b) Backward communication time:  3 * [ 2 (N_DP-1)/N_DP * (2 D D_FF) / W ]"
        "  = 12 (N_DP-1) D D_FF / (N_DP W)"
    )
    print(
        "Justification: we ring all-reduce each of the three FP16 gradient tensors of D·D_FF elements "
        "(W1, W2, W3)."
    )
    print("(c) Compute-bound while  N_DP ≤ 1 + (B W) / C")
    print(
        "Justification: equate 12(N_DP-1)D D_FF/(N_DP W) with 12 B D D_FF/(N_DP C) "
        "and cancel 12 D D_FF/N_DP."
    )

    batch_size = 32.0
    d_model = 4096.0
    d_ff = 16384.0
    num_dp_ranks = 64
    bandwidth = 300e9
    compute_flops_per_s = 989e12

    flops = backward_flops(batch_size=batch_size, d_model=d_model, d_ff=d_ff, num_data_parallel_ranks=num_dp_ranks)
    compute_time = flops / compute_flops_per_s
    comm_time = backward_comm_seconds(
        d_model=d_model, d_ff=d_ff, num_data_parallel_ranks=num_dp_ranks, bandwidth_b_per_s=bandwidth
    )
    max_ranks = max_compute_bound_num_ranks(
        batch_size=batch_size, compute_flops_per_s=compute_flops_per_s, bandwidth_b_per_s=bandwidth
    )
    print("\nNumeric spot-check (illustrative):")
    print(f"  backward_flops≈{flops:.3e}, T_comp≈{compute_time * 1e3:.3f} ms")
    print(f"  T_comm≈{comm_time * 1e3:.3f} ms, N_cap≈{max_ranks:.2f}")


if __name__ == "__main__":
    main()
