from pathlib import Path

import modal

from cs336_systems.profiling_benchmarking.benchmarking_script import GPU
from cs336_systems.profiling_benchmarking.nsys_profile import profile_image, run_nsys_subprocess


WARMUP_STEPS = 5
TIMED_STEPS = 1
BATCH_SIZE = 4
CONTEXT_LENGTHS = [256, 1024, 4096]
METHODS = ["f", "fb", "fbo"]
OUTPUT_DIR = Path("profiles/nsys_profile_experiment_1")

MODEL_SIZES = [
    {
        "name": "small",
        "vocab_size": 10_000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    {
        "name": "medium",
        "vocab_size": 10_000,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
]

app = modal.App("cs336-systems-nsys-profile-experiment-1")


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_csv_ints(value: str) -> list[int]:
    return [int(item) for item in parse_csv(value)]


def selected_model_configs(model_names: list[str]) -> list[dict]:
    configs = {config["name"]: config for config in MODEL_SIZES}
    return [configs[name] for name in model_names]


def profile_output_name(model_name: str, context_length: int, method: str) -> str:
    return f"{model_name}_ctx{context_length}_{method}"


@app.function(image=profile_image, gpu=GPU, timeout=7200)
def run_profile_remote(
    config: dict,
    context_length: int,
    method: str,
    warmup_steps: int,
    timed_steps: int,
    batch_size: int,
    gpu_metrics_device: int,
    include_backtraces: bool,
) -> tuple[str, bytes, str, str]:
    report_path, result = run_nsys_subprocess(
        output_dir=Path("/tmp/nsys_profile_experiment_1"),
        output_name=profile_output_name(config["name"], context_length, method),
        warmup_steps=warmup_steps,
        timed_steps=timed_steps,
        method=method,
        gpu_metrics_device=gpu_metrics_device,
        include_backtraces=include_backtraces,
        batch_size=batch_size,
        vocab_size=config["vocab_size"],
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    return report_path.name, report_path.read_bytes(), result.stdout, result.stderr


@app.local_entrypoint()
def main(
    models: str = "small,medium",
    context_lengths: str = "256,1024,4096",
    methods: str = "f,fb,fbo",
    warmup_steps: int = WARMUP_STEPS,
    timed_steps: int = TIMED_STEPS,
    batch_size: int = BATCH_SIZE,
    output_dir: str = str(OUTPUT_DIR),
    gpu_metrics_device: int = 0,
    include_backtraces: bool = False,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for config in selected_model_configs(parse_csv(models)):
        for context_length in parse_csv_ints(context_lengths):
            for method in parse_csv(methods):
                print(f"Profiling {config['name']} context={context_length} method={method}")
                report_name, report_bytes, stdout, stderr = run_profile_remote.remote(
                    config=config,
                    context_length=context_length,
                    method=method,
                    warmup_steps=warmup_steps,
                    timed_steps=timed_steps,
                    batch_size=batch_size,
                    gpu_metrics_device=gpu_metrics_device,
                    include_backtraces=include_backtraces,
                )

                report_path = output_path / report_name
                report_path.write_bytes(report_bytes)
                (output_path / f"{report_path.stem}.stdout.txt").write_text(stdout)
                (output_path / f"{report_path.stem}.stderr.txt").write_text(stderr)
                print(f"Wrote {report_path}")
