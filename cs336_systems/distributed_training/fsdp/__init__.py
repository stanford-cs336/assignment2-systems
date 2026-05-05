"""Fully-sharded data parallel (FSDP): shard ``Linear`` / ``Embedding`` weights; norms stay replicated."""

from cs336_systems.distributed_training.fsdp.fully_sharded_parallel import FullyShardedDataParallel

__all__ = ["FullyShardedDataParallel"]
