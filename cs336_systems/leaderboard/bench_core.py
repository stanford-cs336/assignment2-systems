from __future__ import annotations

import json
import os
from typing import Any

import torch
import torch.distributed as dist
from cs336_basics.model import BasicsTransformerLM, TransformerBlock

from cs336_systems.distributed_training.fsdp import FullyShardedDataParallel
from cs336_systems.leaderboard.chunked_ce import chunked_lm_head_cross_entropy_loss
from cs336_systems.leaderboard.config import LEADERBOARD_8B, model_kwargs
from cs336_systems.leaderboard.flash_sdpa_patch import install_flash_sdpa


def _grad_checkpoint_enabled() -> bool:
    v = os.environ.get("LEADERBOARD_GRAD_CKPT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _ce_microbatch_size(batch_size: int) -> int:
    raw = os.environ.get("LEADERBOARD_CE_MICROBATCH", "1").strip()
    return max(1, min(batch_size, int(raw)))


def _vocab_chunk_rows() -> int:
    return max(256, int(os.environ.get("LEADERBOARD_CE_CHUNK", "4096")))


def _datacenter_matmul_tuning() -> None:
    if not torch.cuda.is_available():
        return
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def _wrap_block_with_checkpoint(forward_fn):
    import torch.utils.checkpoint as ckpt

    def checkpointed(x: torch.Tensor) -> torch.Tensor:
        return ckpt.checkpoint(forward_fn, x, use_reentrant=False)

    return checkpointed


def _checkpoint_every_transformer_block(module: BasicsTransformerLM) -> None:
    if not _grad_checkpoint_enabled():
        return
    for block in module.layers:
        if isinstance(block, TransformerBlock):
            block.forward = _wrap_block_with_checkpoint(block.forward)  # type: ignore[method-assign]


def _fsdp_sharded_unit_index(fsdp: FullyShardedDataParallel, submodule: torch.nn.Module) -> int:
    for i, unit in enumerate(fsdp._units):
        if unit is submodule:
            return i
    raise RuntimeError(f"{submodule!r} is not an FSDP sharded unit")


def _forward_to_ln_final(model: BasicsTransformerLM, token_ids: torch.Tensor) -> torch.Tensor:
    hidden = model.token_embeddings(token_ids)
    for block in model.layers:
        hidden = block(hidden)
    return model.ln_final(hidden)


def _chunked_ce_loss(
    fsdp: FullyShardedDataParallel,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    *,
    lm_head_idx: int,
    chunk_rows: int,
) -> torch.Tensor:
    lm_head = fsdp.module.lm_head
    fsdp._apply_forward_gather(lm_head, lm_head_idx)

    def loss_from_weight(w: torch.Tensor) -> torch.Tensor:
        return chunked_lm_head_cross_entropy_loss(hidden, w, targets, chunk_rows)

    loss = fsdp._run_forward_with_gathered_weight(lm_head, lm_head_idx, loss_from_weight)
    fsdp._launch_prefetch(lm_head_idx + 2, fsdp._fwd_work, fsdp._fwd_bufs)
    return loss


def _adamw(parameters, **kwargs: Any) -> torch.optim.AdamW:
    for extra in ({"fused": True}, {"foreach": True}, {}):
        try:
            return torch.optim.AdamW(parameters, **extra, **kwargs)
        except (TypeError, ValueError):
            continue
    raise RuntimeError("AdamW() should always be constructible")


def _log_rank0(rank: int, message: str) -> None:
    if rank == 0:
        print(f"[leaderboard rank 0] {message}", flush=True)


def run_benchmark_after_dist_init(
    *,
    rank: int,
    world_size: int,
    device: torch.device,
    bench_warmup: int,
    bench_rep: int,
) -> dict[str, Any] | None:
    install_flash_sdpa()
    _datacenter_matmul_tuning()
    _log_rank0(rank, f"NCCL rank ready (world_size={world_size})")

    torch.manual_seed(0)
    if torch.cuda.is_bf16_supported():
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True

    batch_size = LEADERBOARD_8B["batch_size"]
    vocab_size = LEADERBOARD_8B["vocab_size"]
    seq_len = LEADERBOARD_8B["context_length"]

    base_model = BasicsTransformerLM(**model_kwargs()).to(device=device, dtype=torch.float32)
    _log_rank0(rank, "model allocated; applying gradient checkpointing …")
    _checkpoint_every_transformer_block(base_model)
    model = FullyShardedDataParallel(base_model, compute_dtype=torch.bfloat16)
    _log_rank0(rank, "FSDP ready")

    optimizer = _adamw(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    microbatch_size = _ce_microbatch_size(batch_size)
    chunk_rows = _vocab_chunk_rows()
    lm_head_idx = _fsdp_sharded_unit_index(model, model.module.lm_head)

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, batch_size, microbatch_size):
            model._reset_prefetch_state()
            end = min(start + microbatch_size, batch_size)
            label_batch = labels[start:end]
            target_batch = targets[start:end]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = _forward_to_ln_final(model.module, label_batch)
                loss = _chunked_ce_loss(model, hidden, target_batch, lm_head_idx=lm_head_idx, chunk_rows=chunk_rows)
            loss.backward()
        model.finish_gradient_synchronization()
        optimizer.step()

    _log_rank0(rank, "running first train_step (may trigger compilation) …")
    if rank == 0:
        torch.cuda.synchronize()
    dist.barrier()
    train_step()
    dist.barrier()
    if rank == 0:
        torch.cuda.synchronize()
    _log_rank0(rank, f"warm-up done; benchmarking with warmup={bench_warmup} rep={bench_rep}")

    import triton.testing

    median_ms = float(
        triton.testing.do_bench(
            train_step,
            warmup=bench_warmup,
            rep=bench_rep,
            return_mode="median",
        )
    )
    _log_rank0(rank, f"median_train_step_ms={median_ms:.4f}")

    metrics: dict[str, Any] = {
        "median_train_step_ms": median_ms,
        "peak_mem_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "bench_warmup": bench_warmup,
        "bench_rep": bench_rep,
    }

    dist.barrier()
    dist.destroy_process_group()

    return metrics if rank == 0 else None


def torchrun_main() -> None:
    bench_warmup = int(os.environ.get("LEADERBOARD_BENCH_WARMUP", "25"))
    bench_rep = int(os.environ.get("LEADERBOARD_BENCH_REP", "80"))

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    try:
        dist.init_process_group("nccl", device_id=device)
    except TypeError:
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    out = run_benchmark_after_dist_init(
        rank=rank,
        world_size=dist.get_world_size(),
        device=device,
        bench_warmup=bench_warmup,
        bench_rep=bench_rep,
    )
    if rank == 0 and out is not None:
        path = os.environ.get("LEADERBOARD_RESULT_JSON", "/tmp/leaderboard_result.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2), flush=True)
