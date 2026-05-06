"""Modal app: nsys profiles for DDP (naive/overlap) and FSDP on 2× B200."""
from __future__ import annotations

import glob
import subprocess
import sys
from pathlib import Path

import modal

_WORKSPACE = "/workspace"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04", add_python="3.12")
    .run_commands(
        "apt-get update && "
        "apt-get install -y $(apt-cache search '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -n 1) && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install("torch~=2.11.0", "einops", "einx", "jaxtyping")
    .env({"PYTHONPATH": f"{_WORKSPACE}:{_WORKSPACE}/cs336-basics"})
    .add_local_python_source("cs336_systems")
    .add_local_python_source("cs336_basics")
)

GPU = "B200:2"

app = modal.App("cs336-systems-nsys-ddp-fsdp-profile")

DDP_OUTPUT_DIR = Path("profiles/nsys_ddp")
FSDP_OUTPUT_DIR = Path("profiles/nsys_fsdp")


def _nsys_profile_rank0(output_base: str, worker_module: str, extra_args: list[str], master_port: int) -> list[tuple[str, bytes]]:
    env = {
        **__import__("os").environ,
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(master_port),
        "WORLD_SIZE": "2",
    }

    rank1_env = {**env, "RANK": "1", "LOCAL_RANK": "1"}
    rank1_proc = subprocess.Popen(
        [sys.executable, "-m", worker_module, *extra_args],
        env=rank1_env,
    )

    rank0_env = {**env, "RANK": "0", "LOCAL_RANK": "0"}
    nsys_cmd = [
        "nsys", "profile",
        "--force-overwrite=true",
        "--trace=cuda,nccl,nvtx",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        f"--output={output_base}",
        "--",
        sys.executable, "-m", worker_module,
        *extra_args,
    ]
    subprocess.run(nsys_cmd, check=True, env=rank0_env)
    rank1_proc.wait()

    files = sorted(glob.glob(f"{output_base}*.nsys-rep"))
    return [(Path(f).name, Path(f).read_bytes()) for f in files]


@app.function(image=image, gpu=GPU, timeout=3600)
def run_ddp_profile_remote(mode: str) -> list[tuple[str, bytes]]:
    output_base = f"/tmp/nsys_ddp/ddp_{mode}"
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    return _nsys_profile_rank0(
        output_base,
        "cs336_systems.distributed_training.modal.nsys_ddp_worker",
        ["--mode", mode],
        master_port=29500,
    )


@app.function(image=image, gpu=GPU, timeout=3600)
def run_fsdp_profile_remote() -> list[tuple[str, bytes]]:
    output_base = "/tmp/nsys_fsdp/fsdp_xl"
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    return _nsys_profile_rank0(
        output_base,
        "cs336_systems.distributed_training.modal.nsys_fsdp_worker",
        [],
        master_port=29501,
    )


@app.local_entrypoint()
def main(ddp: bool = True, fsdp: bool = True) -> None:
    if ddp:
        for mode in ["naive", "overlap"]:
            print(f"Profiling DDP mode={mode} ...")
            DDP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for name, data in run_ddp_profile_remote.remote(mode):
                path = DDP_OUTPUT_DIR / name
                path.write_bytes(data)
                print(f"  Wrote {path}")

    if fsdp:
        print("Profiling FSDP ...")
        FSDP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for name, data in run_fsdp_profile_remote.remote():
            path = FSDP_OUTPUT_DIR / name
            path.write_bytes(data)
            print(f"  Wrote {path}")
