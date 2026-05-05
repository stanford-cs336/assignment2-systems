"""Idealized ring collective timings from assignment §8.1.

Per device egress bandwidth ``W`` (bytes/s).  For a *full* payload of ``S`` bytes:

- Ring all-gather:  ``(N - 1) / N * S / W``
- Ring reduce-scatter: same as all-gather
- Ring all-reduce (RS + AG): ``2 (N - 1) / N * S / W``

The alternate all-reduce in §8.1 Problem ``alternate_ring_all_reduce`` sends a *full* ``S``-byte
tensor each round; see ``alternate_ring_allreduce_time``.
"""

from __future__ import annotations


def ring_all_gather_seconds(*, n_ranks: int, payload_bytes: float, bandwidth_b_per_s: float) -> float:
    if n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    if n_ranks == 1:
        return 0.0
    return (n_ranks - 1) / n_ranks * payload_bytes / bandwidth_b_per_s


def ring_reduce_scatter_seconds(*, n_ranks: int, payload_bytes: float, bandwidth_b_per_s: float) -> float:
    return ring_all_gather_seconds(n_ranks=n_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth_b_per_s)


def ring_all_reduce_seconds(*, n_ranks: int, payload_bytes: float, bandwidth_b_per_s: float) -> float:
    return 2.0 * ring_all_gather_seconds(n_ranks=n_ranks, payload_bytes=payload_bytes, bandwidth_b_per_s=bandwidth_b_per_s)


def alternate_ring_allreduce_seconds(*, n_ranks: int, payload_bytes: float, bandwidth_b_per_s: float) -> float:
    """Algorithm from §8.1: each of ``N - 1`` steps ships one full ``S``-byte tensor."""
    if n_ranks < 1:
        raise ValueError("n_ranks must be >= 1")
    if n_ranks == 1:
        return 0.0
    return (n_ranks - 1) * payload_bytes / bandwidth_b_per_s


def fp16_tensor_bytes(num_elems: int) -> float:
    return 2.0 * num_elems
