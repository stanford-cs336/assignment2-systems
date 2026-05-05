# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

from cs336_systems.parallelism.ring_collectives import ring_all_gather_seconds, ring_all_reduce_seconds


def forward_flops(
    *, batch_size: float, d_model: float, d_ff: float, num_fsdp_ranks: int, num_tp_ranks: int
) -> float:
    return 6.0 * batch_size * d_model * d_ff / (num_fsdp_ranks * num_tp_ranks)


def fsdp_forward_gather_time(
    *, d_model: float, d_ff: float, num_fsdp_ranks: int, num_tp_ranks: int, bandwidth_b_per_s: float
) -> float:
    num_elements = d_model * (d_ff / num_tp_ranks)
    payload_bytes = 2.0 * num_elements
    time_one_gather = ring_all_gather_seconds(
        n_ranks=num_fsdp_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )
    return 3.0 * time_one_gather


def tp_forward_allreduce_time(
    *, batch_size: float, d_model: float, num_fsdp_ranks: int, num_tp_ranks: int, bandwidth_b_per_s: float
) -> float:
    payload_bytes = 2.0 * (batch_size / num_fsdp_ranks) * d_model
    return ring_all_reduce_seconds(
        n_ranks=num_tp_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth_b_per_s
    )


def overlapped_max_total_ranks(
    *, batch_size: float, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float
) -> float:
    return 1.5 * batch_size * d_ff * bandwidth_b_per_s * bandwidth_b_per_s / (compute_flops_per_s * compute_flops_per_s)


def serial_max_total_ranks(
    *, batch_size: float, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float
) -> float:
    return 0.375 * batch_size * d_ff * bandwidth_b_per_s * bandwidth_b_per_s / (compute_flops_per_s * compute_flops_per_s)


def is_compute_bound_serial(
    *, batch_size: float, d_model: float, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float,
    num_fsdp_ranks: int, num_tp_ranks: int
) -> bool:
    compute_time = forward_flops(
        batch_size=batch_size, d_model=d_model, d_ff=d_ff, num_fsdp_ranks=num_fsdp_ranks, num_tp_ranks=num_tp_ranks
    ) / compute_flops_per_s
    fsdp_gather_time = fsdp_forward_gather_time(
        d_model=d_model, d_ff=d_ff, num_fsdp_ranks=num_fsdp_ranks, num_tp_ranks=num_tp_ranks,
        bandwidth_b_per_s=bandwidth_b_per_s
    )
    tp_allreduce_time = tp_forward_allreduce_time(
        batch_size=batch_size, d_model=d_model, num_fsdp_ranks=num_fsdp_ranks, num_tp_ranks=num_tp_ranks,
        bandwidth_b_per_s=bandwidth_b_per_s
    )
    return (fsdp_gather_time + tp_allreduce_time) <= compute_time


def search_max_product_serial(
    *, batch_size: float, d_model: float, d_ff: float, compute_flops_per_s: float, bandwidth_b_per_s: float,
    grid_max: int = 256
):
    best_product = 0
    best_num_fsdp = 0
    best_num_tp = 0
    for num_fsdp in range(2, grid_max + 1):
        for num_tp in range(2, grid_max + 1):
            if is_compute_bound_serial(
                batch_size=batch_size, d_model=d_model, d_ff=d_ff,
                compute_flops_per_s=compute_flops_per_s, bandwidth_b_per_s=bandwidth_b_per_s,
                num_fsdp_ranks=num_fsdp, num_tp_ranks=num_tp,
            ):
                total_ranks = num_fsdp * num_tp
                if total_ranks > best_product:
                    best_product = total_ranks
                    best_num_fsdp = num_fsdp
                    best_num_tp = num_tp
    return best_product, best_num_fsdp, best_num_tp


def main() -> None:
    d_model = 4096.0
    d_ff = 16384.0
    batch_size = 256.0
    bandwidth = 300e9
    compute_flops_per_s = 3e12

    print("§8.5 fsdp_tp_calcs")
    print("(a) Forward FLOPs per device:  6 B D D_FF / (N_FSDP N_TP)")
    print("Justification: three local matmuls sized by batch B/N_FSDP and hid D_FF/N_TP (Eqs. 55–58).")
    print(
        "(b) Overlapped forward comm time:  max(T_FSDP-axis, T_TP-axis) with\n"
        "     T_FSDP = 3 * ring_all_gather on FP16 tensor of D*(D_FF/N_TP) elems,\n"
        "     T_TP   = ring_all_reduce on FP16 tensor (B/N_FSDP, D)."
    )
    print(
        "(c) With overlap, balance collectives via T_FSDP ≈ T_TP, then pin T_FSDP = T_comp to obtain\n"
        "     N_FSDP ≈ 1 + (B W)/C and N_TP ≈ 1 + (3 D_FF W)/(2 C) in the\n"
        "     large-rank limit, hence\n"
        "       N = N_TP N_FSDP ≈ (3 B D_FF W^2) / (2 C^2)\n"
        "     up to the usual +1 ring corrections."
    )
    max_overlap = overlapped_max_total_ranks(
        batch_size=batch_size, d_ff=d_ff, compute_flops_per_s=compute_flops_per_s, bandwidth_b_per_s=bandwidth
    )
    max_serial = serial_max_total_ranks(
        batch_size=batch_size, d_ff=d_ff, compute_flops_per_s=compute_flops_per_s, bandwidth_b_per_s=bandwidth
    )
    print(f"\nAnalytic scale check (illustrative constants): N_overlap ≈ {max_overlap:.3e} ; N_serial ≈ {max_serial:.3e}")

    best_product, best_num_fsdp, best_num_tp = search_max_product_serial(
        batch_size=batch_size, d_model=d_model, d_ff=d_ff,
        compute_flops_per_s=compute_flops_per_s, bandwidth_b_per_s=bandwidth,
        grid_max=128,
    )
    if best_product == 0:
        grid_msg = "no pair with N_FSDP,N_TP>=2 in grid (try larger batch or slower C in the demo)."
    else:
        grid_msg = f"max N≈{best_product} at N_FSDP={best_num_fsdp}, N_TP={best_num_tp}"
    print(
        f"(d) Serial comm (grid search up to 128×128): {grid_msg}. "
        "Large-rank closed form: N ≈ (3 B D_FF W²)/(8 C²) when T_FSDP = T_TP and T_FSDP + T_TP = T_comp."
    )


if __name__ == "__main__":
    main()
