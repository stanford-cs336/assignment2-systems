import csv
import gc
from pathlib import Path

import modal
import torch

from cs336_systems.profiling_benchmarking.benchmarking_script import (
    GPU,
    ModelConfigDict,
    capture_memory_snapshot,
    configure_fp32_precision,
    get_random_batch,
    image,
    initialize_model_from_config,
    DTYPE,
)

app = modal.App("cs336-systems-memory-profile")

OUTPUT_DIR = Path("profiles/memory_viz")
BATCH_SIZE = 4
FBO_LONG_CTX_BATCH = 2
LONG_CTX_THRESHOLD = 2048
WARMUP_STEPS = 2

_XL_BASE: dict[str, int] = {
    "vocab_size": 10_000,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
}


def _xl_config(context_length: int) -> ModelConfigDict:
    return {
        "vocab_size": _XL_BASE["vocab_size"],
        "context_length": context_length,
        "d_model": _XL_BASE["d_model"],
        "num_layers": _XL_BASE["num_layers"],
        "num_heads": _XL_BASE["num_heads"],
        "d_ff": _XL_BASE["d_ff"],
    }


def _batch_size(context_length: int, method: str) -> int:
    if method == "fbo" and context_length >= LONG_CTX_THRESHOLD:
        return FBO_LONG_CTX_BATCH
    return BATCH_SIZE


def _tag(context_length: int, method: str, use_bf16: bool, batch_size: int) -> str:
    prec = "bf16" if use_bf16 else "fp32"
    b = f"_b{batch_size}" if batch_size != BATCH_SIZE else ""
    return f"xl_ctx{context_length}_{method}_{prec}{b}"


@app.function(image=image, gpu=GPU, timeout=3600, retries=0)
def run_memory_snapshot_remote(
    model_config: ModelConfigDict,
    batch_size: int,
    method: str,
    warmup_steps: int,
    use_bf16_autocast: bool,
) -> tuple[bytes, int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    configure_fp32_precision()
    model = initialize_model_from_config(model_config).cuda().to(DTYPE)
    batch = get_random_batch(
        batch_size=batch_size,
        vocab_size=model_config["vocab_size"],
        context_length=model_config["context_length"],
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters())
    try:
        return capture_memory_snapshot(
            model,
            batch,
            method,
            optimizer,
            warmup_steps,
            use_bf16_autocast=use_bf16_autocast,
        )
    finally:
        del model
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@app.local_entrypoint()
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    context_lengths = (128, 2048)
    methods = ("f", "fbo")
    precisions = (True, False)  # BF16 first: FP32 OOM poisons the allocator for subsequent BF16 runs

    peak_rows: list[dict[str, str | int | float]] = []
    bytes_per_mib = 1024**2

    for ctx in context_lengths:
        for method in methods:
            for use_bf16 in precisions:
                batch_size = _batch_size(ctx, method)
                tag = _tag(ctx, method, use_bf16, batch_size)
                print(f"Memory snapshot: {tag} (batch={batch_size}, warmup={WARMUP_STEPS})")
                cfg = _xl_config(ctx)
                pickle_path = OUTPUT_DIR / f"{tag}.pickle"
                row = {
                    "context_length": ctx,
                    "batch_size": batch_size,
                    "method": method,
                    "bf16_autocast": int(use_bf16),
                    "peak_allocated_mib": "",
                    "peak_reserved_mib": "",
                    "pickle": str(pickle_path),
                    "error": "",
                }
                try:
                    snapshot_bytes, peak_alloc, peak_reserved = run_memory_snapshot_remote.remote(
                        cfg,
                        batch_size,
                        method,
                        WARMUP_STEPS,
                        use_bf16,
                    )
                    pickle_path.write_bytes(snapshot_bytes)
                    row["peak_allocated_mib"] = f"{peak_alloc / bytes_per_mib:.4f}"
                    row["peak_reserved_mib"] = f"{peak_reserved / bytes_per_mib:.4f}"
                except Exception as exc:
                    row["error"] = f"{type(exc).__name__}: {exc}"
                    print(f"FAILED {tag}: {row['error']}")

                peak_rows.append(row)

    csv_path = OUTPUT_DIR / "peaks.csv"
    fieldnames = list(peak_rows[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(peak_rows)

    ok = sum(1 for r in peak_rows if not r["error"])
    print(f"Wrote {ok}/{len(peak_rows)} snapshots under {OUTPUT_DIR.resolve()}")
    print(f"Peak summary: {csv_path.resolve()}")
