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
RESULTS_PATH = Path("tables/benchmarking_script_results_1.tex")
app = modal.App("cs336-systems-benchmarking-script-experiment-1")
TimingSummary = tuple[float | None, float | None]

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


def initialize_model_from_config(config: dict) -> torch.nn.Module:
    return initialize_model(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).cuda().to(DTYPE)


def mean_and_std(times: list[float]) -> TimingSummary:
    if len(times) == 1:
        return mean(times), 0.0
    return mean(times), stdev(times)


def benchmark_model_size(config: dict) -> dict[str, TimingSummary]:
    batch = get_random_batch(
        batch_size=BATCH_SIZE,
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
    ).cuda()

    method_times: dict[str, list[float] | None] = {}
    for method in ["f", "fb", "fbo"]:
        print(f"Benchmarking {config['name']} method={method}")
        model = initialize_model_from_config(config)
        optimizer = torch.optim.AdamW(model.parameters())
        try:
            method_times[method] = benchmark_method(model, batch, method, optimizer, WARMUP_STEPS, TIMED_STEPS)
        except torch.OutOfMemoryError:
            method_times[method] = None
            print(f"OOM while benchmarking {config['name']} method={method}")
            break
        finally:
            del model
            del optimizer
            torch.cuda.empty_cache()

    forward_times = method_times.get("f")
    forward_backward_times = method_times.get("fb")
    full_step_times = method_times.get("fbo")
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
        "forward": mean_and_std(forward_times) if forward_times else (None, None),
        "backward": mean_and_std(backward_times) if backward_times else (None, None),
        "optimizer": mean_and_std(optimizer_times) if optimizer_times else (None, None),
    }


@app.function(image=image, gpu=GPU, timeout=7200)
def run_experiment_remote() -> dict[str, dict[str, tuple[float, float]]]:
    configure_fp32_precision()
    results = {}
    for config in MODEL_SIZES:
        results[config["name"]] = benchmark_model_size(config)
    return results


def format_timing_value(value: float | None) -> str:
    return "OOM" if value is None else f"{value:.6f}"


def write_latex_table(results: dict[str, dict[str, TimingSummary]]):
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & Forward mean & Forward sd & Backward mean & Backward sd & Optimizer mean & Optimizer sd \\",
        r"\midrule",
    ]

    for model_name, model_results in results.items():
        forward_mean, forward_std = model_results["forward"]
        backward_mean, backward_std = model_results["backward"]
        optimizer_mean, optimizer_std = model_results["optimizer"]
        lines.append(
            f"{model_name} & {format_timing_value(forward_mean)} & {format_timing_value(forward_std)} & "
            f"{format_timing_value(backward_mean)} & {format_timing_value(backward_std)} & "
            f"{format_timing_value(optimizer_mean)} & {format_timing_value(optimizer_std)} \\\\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    RESULTS_PATH.write_text("\n".join(lines) + "\n")


@app.local_entrypoint()
def main():
    results = run_experiment_remote.remote()
    write_latex_table(results)
    print(f"Wrote {RESULTS_PATH}")
