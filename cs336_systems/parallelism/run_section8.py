from __future__ import annotations

from cs336_systems.parallelism import problem_81_alternate_ring_allreduce as p81
from cs336_systems.parallelism import problem_82_data_parallel as p82
from cs336_systems.parallelism import problem_83_fsdp as p83
from cs336_systems.parallelism import problem_84_tensor_parallel as p84
from cs336_systems.parallelism import problem_85_fsdp_tensor_parallel as p85


def main() -> None:
    for fn in (p81.main, p82.main, p83.main, p84.main, p85.main):
        fn()
        print()


if __name__ == "__main__":
    main()
