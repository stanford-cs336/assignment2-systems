# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

from cs336_systems.parallelism.ring_collectives import fp16_tensor_bytes, ring_all_gather_seconds, ring_reduce_scatter_seconds


def forward_flops(*, batch_size: float, d_model: float, d_ff: float, num_fsdp_ranks: int) -> float:
    local_batch = batch_size / num_fsdp_ranks
    return 6.0 * local_batch * d_model * d_ff


def backward_flops(*, batch_size: float, d_model: float, d_ff: float, num_fsdp_ranks: int) -> float:
    local_batch = batch_size / num_fsdp_ranks
    return 12.0 * local_batch * d_model * d_ff


def _weight_bytes_fp16(d_model: float, d_ff: float) -> float:
    return fp16_tensor_bytes(d_model * d_ff)


def forward_comm_seconds(
    *, d_model: float, d_ff: float, num_fsdp_ranks: float, bandwidth_b_per_s: float
) -> float:
    weight_bytes = _weight_bytes_fp16(d_model, d_ff)
    time_one_gather = ring_all_gather_seconds(
        n_ranks=int(num_fsdp_ranks), payload_bytes=weight_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    return 3.0 * time_one_gather


def backward_comm_seconds(
    *, d_model: float, d_ff: float, num_fsdp_ranks: float, bandwidth_b_per_s: float
) -> float:
    weight_bytes = _weight_bytes_fp16(d_model, d_ff)
    time_all_gather = ring_all_gather_seconds(
        n_ranks=int(num_fsdp_ranks), payload_bytes=weight_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    time_reduce_scatter = ring_reduce_scatter_seconds(
        n_ranks=int(num_fsdp_ranks), payload_bytes=weight_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    return 3.0 * (time_all_gather + time_reduce_scatter)


def max_compute_bound_num_ranks(
    *, batch_size: float, compute_flops_per_s: float, bandwidth_b_per_s: float
) -> float:
    return 1.0 + batch_size * bandwidth_b_per_s / compute_flops_per_s


def main() -> None:
    print("§8.3 fsdp_calcs")
    print("(a) Forward FLOPs:  6 (B/N_FSDP) D D_FF ;  Backward:  12 (B/N_FSDP) D D_FF")
    print("Justification: identical per-device matmuls as DP but with batch shard B/N_FSDP.")
    print(
        "(b) Forward comm:  3 * (N-1)/N * (2 D D_FF) / W  = 6 (N-1) D D_FF / (N W)  FP16 bytes."
    )
    print(
        "Backward comm: 3 ring all-gathers + 3 ring reduce-scatters on the same three tensors:"
        "  12 (N-1) D D_FF / (N W)."
    )
    print("Justification: assignment §8.3 lists three all-gathers (fwd) and +3 reduce-scatters (bwd).")
    print("(c) Backward compute-bound while  N_FSDP ≤ 1 + (B W)/C ; forward uses the same bound.")
    print(
        "Justification: compare 12(N-1)D D_FF/(N W) ≤ 12 B D D_FF/(N C) (backward) or "
        "6(N-1)D D_FF/(N W) ≤ 6 B D D_FF/(N C) (forward); both give (N-1)/W ≤ B/C."
    )


if __name__ == "__main__":
    main()
