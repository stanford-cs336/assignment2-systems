"""Run flash-attention pytest targets on a Modal GPU (CUDA + Triton).

Usage (from repo root, after ``modal setup``):

    modal run cs336_systems/gpu_kernels/modal_flash_attention_tests.py

Forward-only Triton tests (default):

    modal run cs336_systems/gpu_kernels/modal_flash_attention_tests.py --pytest-expr test_flash_forward_pass_triton

All Triton flash tests:

    modal run cs336_systems/gpu_kernels/modal_flash_attention_tests.py --pytest-expr 'test_flash_*_triton'
"""

from __future__ import annotations

import modal

# Match ``cs336_systems/profiling_benchmarking/benchmarking_script.py`` (Modal B200 benchmarks in this repo).
GPU = "B200"

_WORKSPACE = "/workspace"

flash_attention_test_image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping", "pytest>=8", "numpy")
    .workdir(_WORKSPACE)
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
    .add_local_dir("tests", remote_path=f"{_WORKSPACE}/tests")
    .add_local_file("pyproject.toml", remote_path=f"{_WORKSPACE}/pyproject.toml")
)

app = modal.App("cs336-systems-flash-attention-tests")


@app.function(image=flash_attention_test_image, gpu=GPU, timeout=1800)
def run_pytest_remote(pytest_expr: str) -> int:
    import torch
    import pytest

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; torch.cuda.is_available() is False")

    args = [
        "tests/test_attention.py",
        "-k",
        pytest_expr,
        "-v",
        "--tb=short",
        "-s",
    ]
    return pytest.main(args)


@app.local_entrypoint()
def main(pytest_expr: str = "test_flash_forward_pass_triton"):
    code = run_pytest_remote.remote(pytest_expr)
    print(f"pytest exit code: {code}")
    if code != 0:
        raise SystemExit(code)
