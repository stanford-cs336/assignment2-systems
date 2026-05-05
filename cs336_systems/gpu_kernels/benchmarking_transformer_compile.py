import csv
from pathlib import Path
from statistics import mean, stdev

import modal
import torch

from cs336_systems.profiling_benchmarking.benchmarking_script import (
    DTYPE,
    GPU,
    benchmark_method,
    configure_fp32_precision,
    get_random_batch,
    image,
    initialize_model,
)

WARMUP_STEPS = 5
TIMED_STEPS = 10
BATCH_SIZE = 4
COMPILE_WARMUP_FB = 6

RESULTS_TEX = Path("tables/benchmarking_transformer_compile.tex")
RESULTS_CSV = Path("tables/benchmarking_transformer_compile.csv")

app = modal.App("cs336-systems-transformer-compile")

MODEL_SIZES = [
    {
        "name": "small",
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    {
        "name": "medium",
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    {
        "name": "large",
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
    },
    {
        "name": "xl",
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10240,
    },
    {
        "name": "7B",
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 11008,
    },
]


def init_model(config: dict) -> torch.nn.Module:
    return initialize_model(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).cuda().to(DTYPE)


def warmup_compiled_backward(model: torch.nn.Module, batch: torch.Tensor, *, steps: int) -> None:
    model.train()
    for _ in range(steps):
        for param in model.parameters():
            param.grad = None
        logits = model(batch)
        loss = logits.mean()
        loss.backward()
        torch.cuda.synchronize()


def benchmark_variant(config: dict, *, compiled: bool) -> dict[str, list[float] | None]:
    batch = get_random_batch(
        batch_size=BATCH_SIZE,
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
    ).cuda()

    seed_model = init_model(config)
    initial_state = seed_model.state_dict()
    del seed_model
    torch.cuda.empty_cache()

    method_times: dict[str, list[float] | None] = {}
    for method in ["f", "fb", "fbo"]:
        model = init_model(config)
        model.load_state_dict(initial_state)
        if compiled:
            model = torch.compile(model)
            warmup_compiled_backward(model, batch, steps=COMPILE_WARMUP_FB)

        optimizer = torch.optim.AdamW(model.parameters())
        try:
            method_times[method] = benchmark_method(model, batch, method, optimizer, WARMUP_STEPS, TIMED_STEPS)
        except torch.OutOfMemoryError:
            method_times[method] = None
            print(f"OOM {'compiled' if compiled else 'vanilla'} {config['name']} method={method}")
            break
        finally:
            del model
            del optimizer
            torch.cuda.empty_cache()

    return method_times


TimingSummary = tuple[float | None, float | None]


def mean_and_std(times: list[float] | None) -> TimingSummary:
    if times is None:
        return None, None
    if len(times) == 1:
        return mean(times), 0.0
    return mean(times), stdev(times)


def summarize_variant_times(times_map: dict[str, list[float] | None]) -> dict[str, TimingSummary]:
    forward_times = times_map.get("f")
    forward_backward_times = times_map.get("fb")
    full_step_times = times_map.get("fbo")
    backward_times = (
        [fb_time - f_time for fb_time, f_time in zip(forward_backward_times, forward_times)]
        if forward_backward_times and forward_times
        else None
    )
    optimizer_times = (
        [fbo_time - fb_time for fbo_time, fb_time in zip(full_step_times, forward_backward_times)]
        if full_step_times and forward_backward_times
        else None
    )
    return {
        "forward": mean_and_std(forward_times),
        "backward": mean_and_std(backward_times),
        "optimizer": mean_and_std(optimizer_times),
    }


def benchmark_model_pair(config: dict) -> tuple[dict[str, TimingSummary], dict[str, TimingSummary]]:
    print(f"Benchmarking {config['name']} vanilla")
    vanilla_times = benchmark_variant(config, compiled=False)
    vanilla_summary = summarize_variant_times(vanilla_times)

    print(f"Benchmarking {config['name']} compiled")
    compiled_times = benchmark_variant(config, compiled=True)
    compiled_summary = summarize_variant_times(compiled_times)

    return vanilla_summary, compiled_summary


def write_tex(rows: list[tuple[str, dict[str, TimingSummary], dict[str, TimingSummary]]]) -> None:
    RESULTS_TEX.parent.mkdir(parents=True, exist_ok=True)

    def fmt(mean_value: float | None) -> str:
        return r"OOM" if mean_value is None else f"${mean_value:.6f}$"

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Fwd.\ vanilla & Fwd.\ compiled & Bwd.\ vanilla & Bwd.\ compiled & Opt.\ vanilla & Opt.\ compiled \\",
        r"\midrule",
    ]
    for model_name, vanilla, compiled in rows:
        vanilla_fwd_mean, _vanilla_fwd_std = vanilla["forward"]
        vanilla_bwd_mean, _vanilla_bwd_std = vanilla["backward"]
        vanilla_opt_mean, _vanilla_opt_std = vanilla["optimizer"]
        compiled_fwd_mean, _compiled_fwd_std = compiled["forward"]
        compiled_bwd_mean, _compiled_bwd_std = compiled["backward"]
        compiled_opt_mean, _compiled_opt_std = compiled["optimizer"]
        lines.append(
            f"{model_name} & {fmt(vanilla_fwd_mean)} & {fmt(compiled_fwd_mean)} & "
            f"{fmt(vanilla_bwd_mean)} & {fmt(compiled_bwd_mean)} & "
            f"{fmt(vanilla_opt_mean)} & {fmt(compiled_opt_mean)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    RESULTS_TEX.write_text("\n".join(lines) + "\n")


def write_csv(rows: list[tuple[str, dict[str, TimingSummary], dict[str, TimingSummary]]]) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "fwd_vanilla_mean", "fwd_vanilla_std",
        "bwd_vanilla_mean", "bwd_vanilla_std",
        "opt_vanilla_mean", "opt_vanilla_std",
        "fwd_compiled_mean", "fwd_compiled_std",
        "bwd_compiled_mean", "bwd_compiled_std",
        "opt_compiled_mean", "opt_compiled_std",
    ]
    with RESULTS_CSV.open("w", newline="") as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fields)
        csv_writer.writeheader()
        for model_name, vanilla, compiled in rows:
            vanilla_fwd_mean, vanilla_fwd_std = vanilla["forward"]
            vanilla_bwd_mean, vanilla_bwd_std = vanilla["backward"]
            vanilla_opt_mean, vanilla_opt_std = vanilla["optimizer"]
            compiled_fwd_mean, compiled_fwd_std = compiled["forward"]
            compiled_bwd_mean, compiled_bwd_std = compiled["backward"]
            compiled_opt_mean, compiled_opt_std = compiled["optimizer"]

            def num(value: float | None) -> str:
                return "" if value is None else str(value)

            csv_writer.writerow(
                {
                    "model": model_name,
                    "fwd_vanilla_mean": num(vanilla_fwd_mean),
                    "fwd_vanilla_std": num(vanilla_fwd_std),
                    "bwd_vanilla_mean": num(vanilla_bwd_mean),
                    "bwd_vanilla_std": num(vanilla_bwd_std),
                    "opt_vanilla_mean": num(vanilla_opt_mean),
                    "opt_vanilla_std": num(vanilla_opt_std),
                    "fwd_compiled_mean": num(compiled_fwd_mean),
                    "fwd_compiled_std": num(compiled_fwd_std),
                    "bwd_compiled_mean": num(compiled_bwd_mean),
                    "bwd_compiled_std": num(compiled_bwd_std),
                    "opt_compiled_mean": num(compiled_opt_mean),
                    "opt_compiled_std": num(compiled_opt_std),
                }
            )


def run_all() -> list[tuple[str, dict[str, TimingSummary], dict[str, TimingSummary]]]:
    configure_fp32_precision()
    rows = []
    for config in MODEL_SIZES:
        vanilla_summary, compiled_summary = benchmark_model_pair(config)
        rows.append((config["name"], vanilla_summary, compiled_summary))
    return rows


@app.function(image=image, gpu=GPU, timeout=7200, retries=0)
def run_remote() -> list[tuple[str, dict[str, TimingSummary], dict[str, TimingSummary]]]:
    return run_all()


@app.local_entrypoint()
def main():
    rows = run_remote.remote()
    write_tex(rows)
    write_csv(rows)
    print(f"Wrote {RESULTS_TEX} and {RESULTS_CSV}")


if __name__ == "__main__":
    rows = run_all()
    write_tex(rows)
    write_csv(rows)
    print(f"Wrote {RESULTS_TEX} and {RESULTS_CSV}")
