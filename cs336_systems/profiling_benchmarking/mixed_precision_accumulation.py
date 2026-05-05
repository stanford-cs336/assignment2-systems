"""Demonstrate FP16 vs FP32 summation error when adding 0.01 one thousand times (assignment §2)."""

from __future__ import annotations

import torch

_NUM_ADDS = 1000


def main() -> None:
    print("Experiment 1: FP32 accumulator, FP32 addends (1000 × 0.01)")
    acc = torch.tensor(0, dtype=torch.float32)
    for _ in range(_NUM_ADDS):
        acc += torch.tensor(0.01, dtype=torch.float32)
    print(f"  sum = {acc.item():.17g}")

    print("Experiment 2: FP16 accumulator, FP16 addends (1000 × 0.01)")
    acc16 = torch.tensor(0, dtype=torch.float16)
    for _ in range(_NUM_ADDS):
        acc16 += torch.tensor(0.01, dtype=torch.float16)
    print(f"  sum = {float(acc16):.17g}")

    print("Experiment 3: FP32 accumulator, FP16 addends (in-place += promotes addend)")
    acc = torch.tensor(0, dtype=torch.float32)
    for _ in range(_NUM_ADDS):
        acc += torch.tensor(0.01, dtype=torch.float16)
    print(f"  sum = {acc.item():.17g}")

    print("Experiment 4: FP32 accumulator, explicit FP16 → FP32 before +=")
    acc = torch.tensor(0, dtype=torch.float32)
    for _ in range(_NUM_ADDS):
        x = torch.tensor(0.01, dtype=torch.float16)
        acc += x.to(dtype=torch.float32)
    print(f"  sum = {acc.item():.17g}")


if __name__ == "__main__":
    main()
