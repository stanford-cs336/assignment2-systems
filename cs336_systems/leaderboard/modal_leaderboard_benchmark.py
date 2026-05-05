# Modal: §9 leaderboard — 2× B200, ``torchrun`` + FSDP + SDPA + chunked LM loss.
#
# **Deploy:**
#   uv run modal deploy -m cs336_systems.leaderboard.modal_leaderboard_benchmark
#
# **Run:** prefer ``--detach`` for long compiles; see comments in ``main``.

from __future__ import annotations

import os
import subprocess
import sys

import modal

from cs336_systems.distributed_training.modal._paths import repo_root

_REPO = repo_root()
_WORKSPACE = "/workspace"

LEADERBOARD_MODAL_APP_NAME = "cs336-systems-leaderboard-8b"
LEADERBOARD_MODAL_REMOTE_FUNCTION_NAME = "run_leaderboard_remote"

GPU = "B200:2"
_STARTUP_TIMEOUT_S = 45 * 60
_EXEC_TIMEOUT_S = 6 * 60 * 60

_TORCHRUN_CMD = [
    sys.executable,
    "-m",
    "torch.distributed.run",
    "--standalone",
    "--nproc_per_node=2",
    "-m",
    "cs336_systems.leaderboard.torchrun_entry",
]
_RESULT_JSON = "/tmp/leaderboard_result.json"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .env({"PYTHONUNBUFFERED": "1"})
    .env({"PYTHONPATH": f"{_WORKSPACE}:{_WORKSPACE}/cs336-basics"})
    .pip_install(
        "torch~=2.11.0",
        "einops",
        "einx",
        "jaxtyping",
        "numpy>=2.4",
        "regex>=2026.3.32",
    )
    .workdir(_WORKSPACE)
    .add_local_dir(str(_REPO / "cs336_systems"), remote_path=f"{_WORKSPACE}/cs336_systems")
    .add_local_dir(str(_REPO / "cs336-basics"), remote_path=f"{_WORKSPACE}/cs336-basics")
)

app = modal.App(LEADERBOARD_MODAL_APP_NAME)


@app.function(
    image=image,
    gpu=GPU,
    startup_timeout=_STARTUP_TIMEOUT_S,
    timeout=_EXEC_TIMEOUT_S,
)
def run_leaderboard_remote(
    bench_warmup: int = 25,
    bench_rep: int = 80,
    grad_ckpt: bool = True,
) -> str:
    import torch

    print(
        f"[leaderboard] Modal bench warmup={bench_warmup} rep={bench_rep} grad_ckpt={grad_ckpt}",
        flush=True,
    )
    if torch.cuda.device_count() < 2:
        raise RuntimeError(f"Need 2 CUDA devices; saw {torch.cuda.device_count()}")

    if not grad_ckpt:
        print(
            "[leaderboard] WARNING: grad_ckpt=False often OOMs at §9 (B=2, T=32k). Continuing.",
            flush=True,
        )

    w = int(os.environ.get("LEADERBOARD_BENCH_WARMUP", str(bench_warmup)))
    r = int(os.environ.get("LEADERBOARD_BENCH_REP", str(bench_rep)))

    print(f"[leaderboard] {' '.join(_TORCHRUN_CMD)}", flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["LEADERBOARD_BENCH_WARMUP"] = str(w)
    env["LEADERBOARD_BENCH_REP"] = str(r)
    env["LEADERBOARD_RESULT_JSON"] = _RESULT_JSON
    env["LEADERBOARD_GRAD_CKPT"] = "1" if grad_ckpt else "0"

    proc = subprocess.run(_TORCHRUN_CMD, cwd=_WORKSPACE, env=env, timeout=None, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"torchrun failed with exit code {proc.returncode}")
    with open(_RESULT_JSON) as f:
        return f.read().strip()


@app.local_entrypoint()
def main(
    bench_warmup: int = 25,
    bench_rep: int = 80,
    grad_ckpt: bool = True,
    wait: bool = True,
) -> None:
    fn = run_leaderboard_remote
    kw = dict(bench_warmup=bench_warmup, bench_rep=bench_rep, grad_ckpt=grad_ckpt)
    if wait:
        print(fn.remote(**kw))
        return

    print(
        "[leaderboard] --no-wait: use `modal run --detach` or the run may cancel when this exits.",
        flush=True,
    )
    call = fn.spawn(**kw)
    print("Leaderboard benchmark enqueued.", flush=True)
    print(call.get_dashboard_url(), flush=True)
    print(f"function_call_id={call.object_id}", flush=True)
