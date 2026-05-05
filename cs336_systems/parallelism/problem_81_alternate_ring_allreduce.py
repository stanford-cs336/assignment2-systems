# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
from __future__ import annotations

from cs336_systems.parallelism.ring_collectives import alternate_ring_allreduce_seconds, ring_all_reduce_seconds


def main() -> None:
    print("§8.1 alternate_ring_all_reduce")
    print("Time:  T = (N - 1) * S / W")
    print(
        "Justification: each of the N − 1 rounds sends one full S-byte tensor from each device "
        "(egress-limited at W), unlike ring RS+AG which only sends S/N per step."
    )
    num_ranks = 8
    payload_bytes = 1024.0 ** 3
    bandwidth = 300e9
    alternate_time = alternate_ring_allreduce_seconds(
        n_ranks=num_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth
    )
    ring_time = ring_all_reduce_seconds(
        n_ranks=num_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth
    )
    print(
        f"\nExample N={num_ranks}, S={payload_bytes:.3e} B, W={bandwidth:.3e} B/s "
        f"→ alternate T={alternate_time * 1e3:.3f} ms, ring T={ring_time * 1e3:.3f} ms"
    )


if __name__ == "__main__":
    main()
