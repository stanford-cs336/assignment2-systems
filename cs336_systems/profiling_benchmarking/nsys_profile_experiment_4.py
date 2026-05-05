import csv
import re
from pathlib import Path


PROFILE_DIR = Path("profiles/nsys_profile_experiment_1")
TIMING_PATHS = [
    Path("tables/nsys_profile_experiment_2_python_timings_small.csv"),
    Path("tables/nsys_profile_experiment_2_python_timings_medium.csv"),
]
CONTEXT_FIT_PATH = Path("tables/nsys_profile_experiment_3_context_fit.csv")
FORWARD_TABLE_PATH = Path("tables/nsys_profile_forward_summary.tex")
KERNEL_TABLE_PATH = Path("tables/nsys_profile_kernel_summary.tex")
CONTEXT_TABLE_PATH = Path("tables/nsys_profile_context_fit_summary.tex")


def latex_escaped(value: str) -> str:
    return value.replace("_", r"\_")


def read_python_timings() -> dict[tuple[str, int, str], float]:
    timings = {}
    for path in TIMING_PATHS:
        with path.open() as timing_file:
            for row in csv.DictReader(timing_file):
                key = (row["model"], int(row["context_length"]), row["method"])
                timings[key] = 1000 * float(row["mean_seconds"])
    return timings


def parse_nvtx_range_ms(profile_text: str, range_name: str) -> float | None:
    match = re.search(rf"\n\s*[0-9.]+\s+([0-9]+)\s+[0-9]+\s+[^\n]*PushPop\s+:{range_name}\s*\n", profile_text)
    if match is None:
        return None
    return int(match.group(1)) / 1e6


def parse_gpu_kernel_stats(profile_text: str) -> list[dict]:
    section = profile_text.split("Executing 'cuda_gpu_kern_sum' stats report", 1)[1]
    kernels = []
    for line in section.splitlines():
        if line.startswith("[7/8]"):
            break
        if not re.match(r"\s*[0-9.]+\s+[0-9]+\s+[0-9]+", line):
            continue
        parts = line.split(None, 8)
        if len(parts) < 9:
            continue
        kernels.append(
            {
                "pct": float(parts[0]),
                "ms": int(parts[1]) / 1e6,
                "instances": int(parts[2]),
                "name": parts[8],
            }
        )
    return kernels


def summarize_profile(stdout_path: Path) -> dict:
    model_name, context_str, method = stdout_path.name.removesuffix(".stdout.txt").split("_")
    profile_text = stdout_path.read_text()
    kernels = parse_gpu_kernel_stats(profile_text)
    total_kernel_ms = sum(kernel["ms"] for kernel in kernels)
    matmul_ms = sum(
        kernel["ms"] for kernel in kernels
        if any(token in kernel["name"] for token in ["cutlass", "gemm", "mma"])
    )
    softmax_ms = sum(
        kernel["ms"] for kernel in kernels
        if any(token in kernel["name"] for token in ["MaxOps", "exp_kernel", "func_wrapp"])
    )
    elementwise_ms = sum(
        kernel["ms"] for kernel in kernels
        if "elementwise_kernel" in kernel["name"] or "vectorized_elementwise" in kernel["name"]
    )
    top_kernel = max(kernels, key=lambda kernel: kernel["ms"])
    return {
        "model": model_name,
        "context": int(context_str.removeprefix("ctx")),
        "method": method,
        "forward_ms": parse_nvtx_range_ms(profile_text, "forward"),
        "backward_ms": parse_nvtx_range_ms(profile_text, "backward"),
        "optimizer_ms": parse_nvtx_range_ms(profile_text, "optimizer_step"),
        "gpu_kernel_ms": total_kernel_ms,
        "matmul_pct": 100 * matmul_ms / total_kernel_ms,
        "softmax_pct": 100 * softmax_ms / total_kernel_ms,
        "elementwise_pct": 100 * elementwise_ms / total_kernel_ms,
        "top_pct": 100 * top_kernel["ms"] / total_kernel_ms,
        "top_instances": top_kernel["instances"],
        "top_kernel": top_kernel["name"],
    }


def kernel_family_label(kernel_name: str) -> str:
    if "cutlass" in kernel_name or "gemm" in kernel_name:
        return "CUTLASS SGEMM"
    if "elementwise" in kernel_name:
        return "PyTorch elementwise"
    if "reduce" in kernel_name:
        return "PyTorch reduction"
    return kernel_name.split("<", maxsplit=1)[0].strip()


def write_forward_table(summaries: list[dict], timings: dict[tuple[str, int, str], float]):
    rows = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Model & Context & Nsight forward GPU kernels (ms) & Python mean (ms) \\",
        r"\midrule",
    ]
    forward_summaries = sorted(
        (summary for summary in summaries if summary["method"] == "f"),
        key=lambda summary: (summary["model"], summary["context"]),
    )
    for summary in forward_summaries:
        python_ms = timings[(summary["model"], summary["context"], "f")]
        rows.append(f"{summary['model']} & {summary['context']} & {summary['gpu_kernel_ms']:.1f} & {python_ms:.1f} \\\\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    FORWARD_TABLE_PATH.write_text("\n".join(rows) + "\n")


def write_kernel_table(summaries: list[dict]):
    profile_by_key = {(summary["model"], summary["context"], summary["method"]): summary for summary in summaries}
    rows = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Model & Context & Top forward kernel & Calls & Forward matmul \% & Forward softmax-ish \% & FBO matmul \% & FBO elementwise \% \\",
        r"\midrule",
    ]
    forward_summaries = sorted(
        (summary for summary in summaries if summary["method"] == "f"),
        key=lambda summary: (summary["model"], summary["context"]),
    )
    for summary in forward_summaries:
        full_step_profile = profile_by_key[(summary["model"], summary["context"], "fbo")]
        rows.append(
            f"{summary['model']} & {summary['context']} & {kernel_family_label(summary['top_kernel'])} & "
            f"{summary['top_instances']} & {summary['matmul_pct']:.1f} & {summary['softmax_pct']:.1f} & "
            f"{full_step_profile['matmul_pct']:.1f} & {full_step_profile['elementwise_pct']:.1f} \\\\"
        )
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    KERNEL_TABLE_PATH.write_text("\n".join(rows) + "\n")


def write_context_table():
    largest_fitting: dict[str, int] = {}
    first_oom: dict[str, int] = {}
    with CONTEXT_FIT_PATH.open() as context_file:
        for row in csv.DictReader(context_file):
            model_name = row["model"]
            context_length = int(row["context_length"])
            if row["fits"] == "True":
                largest_fitting[model_name] = context_length
            elif model_name not in first_oom:
                first_oom[model_name] = context_length

    rows = [
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Model & Largest fitting context & First OOM context \\",
        r"\midrule",
    ]
    for model_name in sorted(largest_fitting):
        rows.append(f"{model_name} & {largest_fitting[model_name]} & {first_oom.get(model_name, -1)} \\\\")
    rows.extend([r"\bottomrule", r"\end{tabular}"])
    CONTEXT_TABLE_PATH.write_text("\n".join(rows) + "\n")


def main():
    summaries = [summarize_profile(stdout_path) for stdout_path in PROFILE_DIR.glob("*.stdout.txt")]
    timings = read_python_timings()
    write_context_table()
    write_forward_table(summaries, timings)
    write_kernel_table(summaries)
    print(f"Wrote {CONTEXT_TABLE_PATH}")
    print(f"Wrote {FORWARD_TABLE_PATH}")
    print(f"Wrote {KERNEL_TABLE_PATH}")


if __name__ == "__main__":
    main()
