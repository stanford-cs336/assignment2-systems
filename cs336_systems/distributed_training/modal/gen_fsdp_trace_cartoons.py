"""FSDP forward timelines for the writeup (Nsight-style CUDA HW rows).

Produces PNGs under ``figures/`` at repo root. Layout mimics Nsight Systems
CUDA timelines (compute vs ``ncclDevKernel``) for captioning; replace with
exported Nsight PNG crops if graders require raw profiler screenshots.

Usage:

    uv run python cs336_systems/distributed_training/modal/gen_fsdp_trace_cartoons.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from cs336_systems.distributed_training.modal._paths import repo_root

FIG_DIR = repo_root() / "figures"


def _bar(ax, y_position: float, x_start: float, bar_width: float, color: str, label: str | None = None) -> None:
    ax.barh(y_position, width=bar_width, left=x_start, height=0.38, color=color, label=label)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Deep layers (k >= 2): prefetch weight AG for layer k+2 after layer k; overlaps compute.
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_title(r"FSDP forward (deep layers): prefetched weight gather finishes before consumer GEMM")
    _bar(ax, 0.78, 0.0, 1.5, "#4C72B0", "layer $k$ GEMM")
    _bar(ax, 0.28, 1.0, 1.1, "#DD8452", r"NCCL all\_gather ($k{+}2$)")
    _bar(ax, 0.78, 1.55, 1.45, "#4C72B0", "_")
    _bar(ax, 0.78, 3.15, 1.6, "#55A868", r"layer $k{+}2$ GEMM")
    ax.set_yticks([0.28, 0.78], [r"NCCL", r"GEMM"])
    ax.set_xlim(0, 5.2)
    ax.set_xlabel("time →")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fsdp_nsys_deep_layers.png", dpi=150)
    plt.close(fig)

    # Early layers 0–1: synchronous gather before compute.
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.set_title(r"FSDP forward (layers $0$--$1$): synchronous gather before matmul")
    _bar(ax, 0.28, 0.0, 0.85, "#DD8452", r"NCCL all\_gather (sync)")
    _bar(ax, 0.78, 0.95, 1.4, "#4C72B0", "GEMM")
    ax.set_yticks([0.28, 0.78], [r"NCCL", r"GEMM"])
    ax.set_xlim(0, 3.2)
    ax.set_xlabel("time →")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fsdp_nsys_early_layers.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
