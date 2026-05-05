"""Leaderboard model/config matching the assignment §9 spec."""

from __future__ import annotations

import torch

# Assignment §9 (8B-class single-node leaderboard)
LEADERBOARD_8B: dict = {
    "vocab_size": 151_936,
    "context_length": 32_768,
    "d_model": 4096,
    "d_ff": 11_008,
    "num_layers": 34,
    "num_heads": 32,
    "batch_size": 2,
    "torch_dtype": torch.bfloat16,
}


def model_kwargs() -> dict:
    config = {**LEADERBOARD_8B}
    config.pop("torch_dtype")
    config.pop("batch_size")
    return config
