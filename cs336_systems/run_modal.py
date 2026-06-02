"""
Modal launcher for the CS336 Assignment 2 nsys benchmark — uv workspace version.

This file lives at: assignment2-systems/cs336_systems/run_modal.py
Project layout it assumes:

    cs336/
    ├── assignment1-basics/        <- your OWN cs336_basics impl (editable dep)
    └── assignment2-systems/       <- project root (has pyproject.toml + uv.lock)
        └── cs336_systems/
            └── run_modal.py       <- this file

Run from the project root:
    modal run cs336_systems/run_modal.py
    modal run cs336_systems/run_modal.py --gpu "L4"

Download the report afterwards and open it in the Nsight Systems GUI:
    modal volume get cs336-profiles train_profile.nsys-rep ./result/
"""

from pathlib import Path

import modal

# --------------------------------------------------------------------------
# Local-only path resolution (runs ONLY locally; see modal.is_local() note).
# This module is imported both locally (to build the image) and in the
# container (to find the function). __file__ differs between the two, so the
# path math must be guarded.
# --------------------------------------------------------------------------

if modal.is_local():
    HERE = Path(__file__).resolve().parent           # .../assignment2-systems/cs336_systems
    PROJECT_ROOT = HERE.parent                        # .../assignment2-systems
    ASSIGNMENT1 = PROJECT_ROOT.parent / "assignment1-basics"  # .../cs336/assignment1-basics

    assert (PROJECT_ROOT / "pyproject.toml").exists(), f"No pyproject.toml in {PROJECT_ROOT}"
    assert (ASSIGNMENT1 / "pyproject.toml").exists(), (
        f"Expected your assignment1-basics package at {ASSIGNMENT1}. "
        "Adjust ASSIGNMENT1 if your repos aren't siblings under cs336/."
    )

    PROJECT_SRC = str(PROJECT_ROOT)
    ASSIGNMENT1_SRC = str(ASSIGNMENT1)
else:
    PROJECT_SRC = "/root/project"
    ASSIGNMENT1_SRC = "/root/assignment1-basics"

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

GPU = "H100!"  # "!" pins the card so Modal won't auto-upgrade to H200 (keeps timings comparable)
CUDA_TAG = "nvidia/cuda:12.4.1-devel-ubuntu22.04"
VENV_PYTHON = "/root/project/.venv/bin/python"

# NVIDIA's CUDA repo does NOT carry nsys; it lives in the separate "devtools"
# repo. The CLI-only package puts `nsys` on PATH and is meant for headless/Docker.
# NOTE: the devtools repo is signed with NVIDIA's LEGACY key (id ...7FA2AF80),
# i.e. 7fa2af80.pub — NOT the newer 3bf863cc.pub used by the CUDA repo.
NSYS_KEY_URL = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub"
NSYS_REPO = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64"

_IGNORE = [
    "**/__pycache__", "*.pyc", ".git", ".venv",
    "result", "data", "*.bin", "*.nsys-rep", "*.sqlite",
]

image = (
    modal.Image.from_registry(CUDA_TAG, add_python="3.12")
    .apt_install("gnupg", "wget")
    # Install Nsight Systems CLI from NVIDIA's devtools repo.
    .run_commands(
        f"wget -qO- {NSYS_KEY_URL} | gpg --dearmor -o /usr/share/keyrings/nvidia-devtools.gpg",
        f"echo 'deb [signed-by=/usr/share/keyrings/nvidia-devtools.gpg] {NSYS_REPO} /' "
        "> /etc/apt/sources.list.d/nvidia-devtools.list",
        "apt-get update && apt-get install -y --no-install-recommends nsight-systems-cli",
        "nsys --version",  # fail the BUILD (not runtime) if nsys isn't on PATH
    )
    .pip_install("uv")
    .add_local_dir(PROJECT_SRC, remote_path="/root/project", copy=True, ignore=_IGNORE)
    .add_local_dir(ASSIGNMENT1_SRC, remote_path="/root/assignment1-basics", copy=True, ignore=_IGNORE)
    .workdir("/root/project")
    .run_commands("uv sync --frozen")  # drop --frozen if it complains the lock is stale
)

app = modal.App("cs336-a2-benchmark", image=image)
volume = modal.Volume.from_name("cs336-profiles", create_if_missing=True)


TRAIN_ARGS = [
    "cs336_systems/train.py",
    "--vocab_size", "10000",
    "--d_model", "256",
    "--num_layers", "4",
    "--num_heads", "8",
    "--d_ff", "768",
    "--context_length", "256",
    "--batch_size", "16",
    "--num_steps", "15",
    "--bm_mode",
    "--warmup", "5",
    "--num_bm", "10",
    "--backward",
]


@app.function(gpu=GPU, volumes={"/output": volume}, timeout=30 * 60)
def profile(use_nsys: bool = True):
    import subprocess

    # Quick sanity check: is CUDA actually visible inside the container?
    check = subprocess.run(
        [VENV_PYTHON, "-c",
         "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
        cwd="/root/project", capture_output=True, text=True,
    )
    print("[cuda-check]", check.stdout.strip(), check.stderr.strip())

    if use_nsys:
        cmd = [
            "nsys", "profile",
            "-o", "/output/train_profile",
            "--force-overwrite", "true",
            "--sample=none",  # CPU sampling needs perf privileges containers lack; CUDA/NVTX/osrt still traced
            "-t", "cuda,cudnn,cublas,nvtx,osrt",
            VENV_PYTHON, *TRAIN_ARGS,
        ]
    else:
        # Bare run (no profiler) to surface train.py's real stdout/stderr and
        # isolate whether the SIGSEGV is from the program itself or from nsys teardown.
        cmd = [VENV_PYTHON, *TRAIN_ARGS]

    proc = subprocess.run(cmd, cwd="/root/project")
    print(f"[exit] return code = {proc.returncode}")

    if use_nsys:
        volume.commit()
        print("Done -> volume 'cs336-profiles': train_profile.nsys-rep")

    # A segfault during interpreter/CUDA teardown AFTER the report is written is
    # usually harmless — the .nsys-rep is already complete. Only treat a failure
    # as fatal if it happened before we got the report.
    if proc.returncode not in (0, -11, 139) :
        raise RuntimeError(f"train.py failed with return code {proc.returncode}")


@app.local_entrypoint()
def main(gpu: str = GPU, no_profile: bool = False):
    # modal run cs336_systems/run_modal.py --no-profile --gpu "L4"
    #   -> bare run, shows train.py output, no nsys (for debugging the 139 segfault)
    # modal run cs336_systems/run_modal.py --gpu "L4"
    #   -> normal profiled run
    profile.with_options(gpu=gpu).remote(use_nsys=not no_profile)
