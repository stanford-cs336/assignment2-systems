# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

from cs336_systems.parallelism.ring_collectives import ring_all_gather_seconds, ring_all_reduce_seconds


def forward_flops(*, batch_size: float, d_model: float, d_ff: float, num_tp_ranks: int) -> float:
    return 6.0 * batch_size * d_model * d_ff / num_tp_ranks


def backward_flops(*, batch_size: float, d_model: float, d_ff: float, num_tp_ranks: int) -> float:
    return 12.0 * batch_size * d_model * d_ff / num_tp_ranks


def forward_comm_seconds(
    *, batch_size: float, d_model: float, d_ff: float, num_tp_ranks: int, bandwidth_b_per_s: float
) -> float:
    activation_bytes = 2.0 * batch_size * d_ff
    time_all_gather = ring_all_gather_seconds(
        n_ranks=num_tp_ranks, payload_bytes=activation_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    output_bytes = 2.0 * batch_size * d_model
    time_all_reduce = ring_all_reduce_seconds(
        n_ranks=num_tp_ranks, payload_bytes=output_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    return 2.0 * time_all_gather + time_all_reduce


def backward_comm_seconds(
    *, batch_size: float, d_model: float, num_tp_ranks: int, bandwidth_b_per_s: float
) -> float:
    input_grad_bytes = 2.0 * batch_size * d_model
    return ring_all_reduce_seconds(
        n_ranks=num_tp_ranks, payload_bytes=input_grad_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )


def max_tp_ranks_forward(*, d_model: float, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float) -> float:
    return 1.0 + (3.0 * d_model * d_ff * bandwidth_b_per_s) / (2.0 * compute_flops_per_s * (d_model + d_ff))


def max_tp_ranks_backward(*, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float) -> float:
    return 1.0 + (3.0 * d_ff * bandwidth_b_per_s) / compute_flops_per_s


def main() -> None:
    print("§8.4 tp_calcs — backward pass (per TP rank i, ⊙ Hadamard)")
    print(
        """
  dz^(i)           = dy @ (W_3^(i))^T
  d x_2^(i)       = dz^(i) ⊙ f(x_1^(i))
  d x_1^(i)       = dz^(i) ⊙ f'(x_1^(i)) ⊙ x_2^(i)
  d W_3^(i)       = (z^(i))^T @ dy
  d W_2^(i)       = x^T @ d x_2^(i)
  d W_1^(i)       = x^T @ d x_1^(i)
  d x_contrib^(i) = d x_1^(i) @ (W_1^(i))^T + d x_2^(i) @ (W_2^(i))^T
  d x             = all_reduce_sum({ d x_contrib^(i) }_{i=0}^{N_TP-1})
"""
    )
    print(
        "(b) Forward FLOPs / device:  6 B D D_FF / N_TP ;  Backward:  12 B D D_FF / N_TP "
        "(same matmul shapes as §8.2 but sharded on D_FF or D_FF/N_TP)."
    )
    print(
        "(c) Forward comm:  2 * [(N_TP-1)/(N_TP W) * 2 B D_FF]  +  2(N_TP-1)/(N_TP W) * 2 B D"
        "  = 4 (N_TP-1) B (D + D_FF) / (N_TP W)."
    )
    print(
        "Backward comm (dominant): one ring all-reduce on d x (FP16 (B,D)):"
        "  4 (N_TP-1) B D / (N_TP W)."
    )
    print("(d) Forward compute-bound while  N_TP ≤ 1 + (3 D D_FF W) / (2 C (D + D_FF))")
    print("    Backward compute-bound while  N_TP ≤ 1 + (3 D_FF W) / C")
    print(
        "Justification: cancel the common B/N_TP from T_comp vs T_comm after equating the §8.1 ring costs."
    )


if __name__ == "__main__":
    main()
