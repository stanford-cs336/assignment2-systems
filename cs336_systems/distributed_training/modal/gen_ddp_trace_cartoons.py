"""Timeline cartoons for DDP writeup (optional Nsight replacement).

Writes PNGs under ``figures/`` at repo root.

Usage:

    uv run python cs336_systems/distributed_training/modal/gen_ddp_trace_cartoons.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from cs336_systems.distributed_training.modal._paths import repo_root

FIG_DIR = repo_root() / "figures"


def _bar(ax, y_position: float, x_start: float, bar_width: float, color: str, label: str) -> None:
    ax.barh(y_position, width=bar_width, left=x_start, height=0.35, color=color, label=label)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.8), sharex=True)

    ax = axes[0]
    ax.set_title("Naive DDP (post-backward blocking all-reduces)")
    _bar(ax, 0.7, 0.0, 5.0, "#4C72B0", "backward")
    _bar(ax, 0.3, 5.0, 1.8, "#DD8452", "NCCL all-reduces")
    ax.set_yticks([0.3, 0.7], ["NCCL", "backward"])
    ax.set_xlim(0, 8)
    ax.set_xlabel("time →")

    ax = axes[1]
    ax.set_title("Overlapping DDP (async per-param during backward)")
    layer_intervals = [(0, 1.2), (1.4, 2.6), (2.8, 4.2)]
    for layer_index, (start, end) in enumerate(layer_intervals):
        _bar(ax, 0.72, start, end - start, "#4C72B0", "backward" if layer_index == 0 else "_")
    comm_intervals = [(1.1, 0.9), (2.5, 0.85), (3.9, 1.1)]
    for comm_index, (start, bar_width) in enumerate(comm_intervals):
        _bar(ax, 0.32, start, bar_width, "#DD8452", "NCCL" if comm_index == 0 else "_")
    ax.set_yticks([0.32, 0.72], ["NCCL", "backward"])
    ax.set_xlim(0, 8)
    ax.set_xlabel("time →")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "ddp_trace_naive_cartoon.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.set_title("Overlapping DDP (conceptual overlap)")
    for start, end in layer_intervals:
        _bar(ax, 0.72, start, end - start, "#4C72B0", "backward")
    for comm_index, (start, bar_width) in enumerate(comm_intervals):
        _bar(ax, 0.32, start, bar_width, "#DD8452", "NCCL" if comm_index == 0 else "_")
    ax.set_yticks([0.32, 0.72], ["NCCL", "backward"])
    ax.set_xlim(0, 8)
    ax.set_xlabel("time →")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "ddp_trace_overlap_cartoon.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
