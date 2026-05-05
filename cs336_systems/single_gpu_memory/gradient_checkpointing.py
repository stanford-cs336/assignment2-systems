import argparse
import gc
from contextlib import nullcontext
from pathlib import Path

import modal
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM

from cs336_systems.profiling_benchmarking.benchmarking_script import GPU, configure_fp32_precision, get_random_batch, image

app = modal.App("cs336-systems-gradient-checkpointing")

OUTPUT_TEX = Path("tables/gradient_checkpoint_peaks.tex")

XL_CONFIG: dict = {
    "vocab_size": 10_000,
    "context_length": 2048,
    "d_model": 2560,
    "num_layers": 32,
    "num_heads": 32,
    "d_ff": 10240,
}


class CheckpointedBasicsTransformerLM(BasicsTransformerLM):
    def __init__(self, *args, segment_blocks: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.segment_blocks = segment_blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embeddings(x)

        if self.segment_blocks is None:
            for layer in self.layers:
                hidden = layer(hidden)
        else:
            layers_per_chunk = self.segment_blocks
            layers_list = list(self.layers)
            if layers_per_chunk < 1:
                raise ValueError("segment_blocks must be >= 1")
            for start in range(0, len(layers_list), layers_per_chunk):
                chunk_layers = layers_list[start : start + layers_per_chunk]

                def run_chunk(inp: torch.Tensor, chunk_layers: tuple[nn.Module, ...] = tuple(chunk_layers)) -> torch.Tensor:
                    result = inp
                    for layer in chunk_layers:
                        result = layer(result)
                    return result

                hidden = checkpoint(run_chunk, hidden, use_reentrant=False)

        hidden = self.ln_final(hidden)
        logits = self.lm_head(hidden)
        return logits


def forward_recursive_checkpointed(
    inp: torch.Tensor,
    layers: nn.ModuleList,
    lo: int,
    hi: int,
) -> torch.Tensor:
    if lo == hi:
        return layers[lo](inp)

    mid = (lo + hi) // 2

    def first_half(x: torch.Tensor) -> torch.Tensor:
        return forward_recursive_checkpointed(x, layers, lo, mid)

    x_mid = checkpoint(first_half, inp, use_reentrant=False)

    def second_half(x: torch.Tensor) -> torch.Tensor:
        return forward_recursive_checkpointed(x, layers, mid + 1, hi)

    return checkpoint(second_half, x_mid, use_reentrant=False)


def one_step_peak_mib(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    *,
    use_bf16_autocast: bool,
) -> float | None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    autocast_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16_autocast else nullcontext()
    try:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context:
            logits = model(batch)
            loss = logits.mean()
        loss.backward()
        torch.cuda.synchronize()
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        return peak_allocated_bytes / (1024**3)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.synchronize()
        return None
    finally:
        del optimizer
        gc.collect()
        torch.cuda.empty_cache()


def warmup_and_measure(
    model: BasicsTransformerLM,
    batch: torch.Tensor,
    *,
    warmup_steps: int,
    use_bf16_autocast: bool,
) -> float | None:
    for _ in range(warmup_steps):
        peak = one_step_peak_mib(model, batch, use_bf16_autocast=use_bf16_autocast)
        if peak is None:
            return None
    torch.cuda.reset_peak_memory_stats()
    return one_step_peak_mib(model, batch, use_bf16_autocast=use_bf16_autocast)


def sweep_segment_sizes(
    segment_sizes: list[int | None],
    *,
    warmup_steps: int,
    use_bf16_autocast: bool,
) -> list[tuple[str, float | None]]:
    configure_fp32_precision()
    rows: list[tuple[str, float | None]] = []
    batch = get_random_batch(
        batch_size=4,
        vocab_size=XL_CONFIG["vocab_size"],
        context_length=XL_CONFIG["context_length"],
    ).cuda()

    for segment_size in segment_sizes:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        label = "none (vanilla)" if segment_size is None else str(segment_size)
        if segment_size is None:
            model = BasicsTransformerLM(**XL_CONFIG).cuda().to(torch.float32)
        else:
            model = CheckpointedBasicsTransformerLM(**XL_CONFIG, segment_blocks=segment_size).cuda().to(torch.float32)

        peak_gib = warmup_and_measure(model, batch, warmup_steps=warmup_steps, use_bf16_autocast=use_bf16_autocast)
        rows.append((label, peak_gib))
        del model
    return rows


def write_results_tex(rows: list[tuple[str, float | None]], *, use_bf16_autocast: bool) -> None:
    OUTPUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    prec = r"BF16 \texttt{autocast}" if use_bf16_autocast else r"FP32"
    lines = [
        r"\begin{tabular}{lrr}",
        r"  \toprule",
        r"  Layers per \texttt{checkpoint} chunk & Peak alloc.\ (GiB) & Notes \\",
        r"  \midrule",
    ]
    for label, peak in rows:
        if peak is None:
            peak_s = "---"
            note = r"OOM"
        else:
            peak_s = f"${peak:.2f}$"
            note = prec
        escaped = label.replace("_", r"\_")
        lines.append(f"  {escaped} & {peak_s} & {note} \\\\")
    lines += [r"  \bottomrule", r"\end{tabular}"]
    OUTPUT_TEX.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_sweep_locally(*, warmup_steps: int, use_bf16_autocast: bool) -> list[tuple[str, float | None]]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script.")
    segment_sizes: list[int | None] = [None, 1, 2, 4, 5, 6, 8, 16, 32]
    return sweep_segment_sizes(segment_sizes, warmup_steps=warmup_steps, use_bf16_autocast=use_bf16_autocast)


@app.function(image=image, gpu=GPU, timeout=3600, retries=0)
def run_remote(*, warmup_steps: int, use_bf16_autocast: bool) -> list[tuple[str, float | None]]:
    return _run_sweep_locally(warmup_steps=warmup_steps, use_bf16_autocast=use_bf16_autocast)


@app.local_entrypoint()
def main():
    rows = run_remote.remote(warmup_steps=2, use_bf16_autocast=True)
    write_results_tex(rows, use_bf16_autocast=True)
    print(f"Wrote {OUTPUT_TEX}")
    for label, peak in rows:
        print(label, peak)


def argparse_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--fp32-matmuls", action="store_true", help="Disable BF16 autocast (may OOM at batch 4, ctx 2048).")
    args = parser.parse_args()

    use_bf16 = not args.fp32_matmuls

    rows = _run_sweep_locally(warmup_steps=args.warmup_steps, use_bf16_autocast=use_bf16)
    write_results_tex(rows, use_bf16_autocast=use_bf16)
    print(f"Wrote {OUTPUT_TEX}")
    for label, peak in rows:
        print(f"{label}: {peak} GiB")


if __name__ == "__main__":
    argparse_main()
