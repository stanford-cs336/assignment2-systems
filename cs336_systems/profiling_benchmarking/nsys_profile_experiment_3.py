from pathlib import Path

import modal
import torch

from cs336_basics.optimizer import AdamW
from cs336_systems.profiling_benchmarking.benchmarking_script import (
    DTYPE,
    GPU,
    configure_fp32_precision,
    get_random_batch,
    image,
    initialize_model,
    run_step,
)
from cs336_systems.profiling_benchmarking.nsys_profile_experiment_1 import BATCH_SIZE, parse_csv, selected_model_configs


MIN_CONTEXT_LENGTH = 256
MAX_CONTEXT_LENGTH = 8192
RESULTS_PATH = Path("tables/nsys_profile_experiment_3_context_fit.csv")
app = modal.App("cs336-systems-nsys-profile-experiment-3")


@app.function(image=image, gpu=GPU, timeout=3600)
def can_fit_context_remote(config: dict, context_length: int, batch_size: int) -> tuple[bool, str]:
    try:
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
        run_step(model, batch, "fbo", optimizer)
        torch.cuda.synchronize()
        return True, ""
    except torch.cuda.OutOfMemoryError as error:
        return False, str(error).splitlines()[0]


def powers_of_two(min_context_length: int, max_context_length: int) -> list[int]:
    context_length = 1
    while context_length < min_context_length:
        context_length *= 2

    values = []
    while context_length <= max_context_length:
        values.append(context_length)
        context_length *= 2
    return values


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["model,context_length,batch_size,fits,error"]
    for row in rows:
        error = row["error"].replace(",", ";")
        lines.append(f"{row['model']},{row['context_length']},{row['batch_size']},{row['fits']},{error}")
    path.write_text("\n".join(lines) + "\n")


@app.local_entrypoint()
def main(
    models: str = "small,medium",
    min_context_length: int = MIN_CONTEXT_LENGTH,
    max_context_length: int = MAX_CONTEXT_LENGTH,
    batch_size: int = BATCH_SIZE,
    results_path: str = str(RESULTS_PATH),
):
    rows = []
    for config in selected_model_configs(parse_csv(models)):
        largest_fit = None
        for context_length in powers_of_two(min_context_length, max_context_length):
            print(f"Testing {config['name']} context={context_length}")
            fits, error = can_fit_context_remote.remote(config=config, context_length=context_length, batch_size=batch_size)
            rows.append(
                {
                    "model": config["name"],
                    "context_length": context_length,
                    "batch_size": batch_size,
                    "fits": fits,
                    "error": error,
                }
            )
            if fits:
                largest_fit = context_length
            else:
                break
        print(f"Largest fitting context for {config['name']}: {largest_fit}")

    write_csv(rows, Path(results_path))
    print(f"Wrote {results_path}")
