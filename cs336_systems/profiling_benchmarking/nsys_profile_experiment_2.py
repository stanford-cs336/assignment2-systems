from pathlib import Path
from statistics import mean, stdev

import modal

from cs336_basics.optimizer import AdamW
from cs336_systems.profiling_benchmarking.benchmarking_script import (
    DTYPE,
    GPU,
    benchmark_method,
    configure_fp32_precision,
    get_random_batch,
    image,
    initialize_model,
)
from cs336_systems.profiling_benchmarking.nsys_profile_experiment_1 import (
    BATCH_SIZE,
    CONTEXT_LENGTHS,
    METHODS,
    WARMUP_STEPS,
    parse_csv,
    parse_csv_ints,
    selected_model_configs,
)


TIMED_STEPS = 10
RESULTS_PATH = Path("tables/nsys_profile_experiment_2_python_timings.csv")
app = modal.App("cs336-systems-nsys-profile-experiment-2")


def mean_and_sd(times: list[float]) -> tuple[float, float]:
    if len(times) == 1:
        return mean(times), 0.0
    return mean(times), stdev(times)


@app.function(image=image, gpu=GPU, timeout=7200)
def run_timing_remote(config: dict, context_length: int, method: str, warmup_steps: int, timed_steps: int, batch_size: int) -> tuple[list[float], float, float]:
    configure_fp32_precision()
    model = initialize_model(
        vocab_size=config["vocab_size"],
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).cuda().to(DTYPE)
    batch = get_random_batch(batch_size=batch_size, vocab_size=config["vocab_size"], context_length=context_length).cuda()
    optimizer = AdamW(model.parameters())

    times = benchmark_method(model, batch, method, optimizer, warmup_steps, timed_steps)
    timing_mean, timing_sd = mean_and_sd(times)
    return times, timing_mean, timing_sd


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["model,context_length,method,batch_size,warmup_steps,timed_steps,mean_seconds,sd_seconds,times_seconds"]
    for row in rows:
        times = " ".join(f"{time:.6f}" for time in row["times"])
        lines.append(
            f"{row['model']},{row['context_length']},{row['method']},{row['batch_size']},"
            f"{row['warmup_steps']},{row['timed_steps']},{row['mean_seconds']:.6f},{row['sd_seconds']:.6f},{times}"
        )
    path.write_text("\n".join(lines) + "\n")


@app.local_entrypoint()
def main(
    models: str = "small,medium",
    context_lengths: str = ",".join(str(context_length) for context_length in CONTEXT_LENGTHS),
    methods: str = ",".join(METHODS),
    warmup_steps: int = WARMUP_STEPS,
    timed_steps: int = TIMED_STEPS,
    batch_size: int = BATCH_SIZE,
    results_path: str = str(RESULTS_PATH),
):
    rows = []
    for config in selected_model_configs(parse_csv(models)):
        for context_length in parse_csv_ints(context_lengths):
            for method in parse_csv(methods):
                print(f"Timing {config['name']} context={context_length} method={method}")
                times, timing_mean, timing_sd = run_timing_remote.remote(
                    config=config,
                    context_length=context_length,
                    method=method,
                    warmup_steps=warmup_steps,
                    timed_steps=timed_steps,
                    batch_size=batch_size,
                )
                rows.append(
                    {
                        "model": config["name"],
                        "context_length": context_length,
                        "method": method,
                        "batch_size": batch_size,
                        "warmup_steps": warmup_steps,
                        "timed_steps": timed_steps,
                        "mean_seconds": timing_mean,
                        "sd_seconds": timing_sd,
                        "times": times,
                    }
                )

    write_csv(rows, Path(results_path))
    print(f"Wrote {results_path}")
