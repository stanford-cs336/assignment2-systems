"""Run ``tests/test_sharded_optimizer.py`` on Modal (CPU + gloo).

Usage:

    modal run cs336_systems/distributed_training/modal/modal_sharded_optimizer_tests.py
"""

from __future__ import annotations

import modal

from cs336_systems.distributed_training.modal._paths import repo_root

_REPO = repo_root()
_WORKSPACE = "/workspace"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping", "pytest>=8", "numpy")
    .env({"PYTHONPATH": f"{_WORKSPACE}:{_WORKSPACE}/cs336-basics"})
    .workdir(_WORKSPACE)
    .add_local_dir(str(_REPO / "cs336_systems"), remote_path=f"{_WORKSPACE}/cs336_systems")
    .add_local_dir(str(_REPO / "cs336-basics"), remote_path=f"{_WORKSPACE}/cs336-basics")
    .add_local_dir(str(_REPO / "tests"), remote_path=f"{_WORKSPACE}/tests")
    .add_local_file(str(_REPO / "pyproject.toml"), remote_path=f"{_WORKSPACE}/pyproject.toml")
)

app = modal.App("cs336-systems-sharded-optimizer-tests")


@app.function(image=image, timeout=1800)
def run_pytest_remote() -> int:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    import pytest
    import torch

    if torch.cuda.is_available():
        raise RuntimeError("Expected CPU-only run")

    return int(pytest.main(["tests/test_sharded_optimizer.py", "-v", "--tb=short", "-s"]))


@app.local_entrypoint()
def main() -> None:
    code = run_pytest_remote.remote()
    print(f"pytest exit code: {code}")
    if code != 0:
        raise SystemExit(code)
