"""``torchrun`` / ``torch.distributed.run`` entry (stable logs under Modal)."""

from __future__ import annotations

from cs336_systems.leaderboard.bench_core import torchrun_main

if __name__ == "__main__":
    torchrun_main()
