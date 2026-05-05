"""Invoke the **deployed** §9 leaderboard Modal function.

Deploy once (or after you change code bundled into the image):

    uv run modal deploy -m cs336_systems.leaderboard.modal_leaderboard_benchmark

Then from any machine with Modal credentials (``modal token new`` / ``MODAL_TOKEN_*``):

    uv run python -m cs336_systems.leaderboard.invoke_deployed
    uv run python -m cs336_systems.leaderboard.invoke_deployed --no-wait

This uses :func:`modal.Function.from_name` — it does **not** re-upload ``cs336_systems`` via
``add_local_dir``; the container runs whatever version was last deployed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from cs336_systems.leaderboard.modal_leaderboard_benchmark import (
    LEADERBOARD_MODAL_APP_NAME,
    LEADERBOARD_MODAL_REMOTE_FUNCTION_NAME,
)


def main() -> None:
    p = argparse.ArgumentParser(description="Call deployed run_leaderboard_remote on Modal.")
    p.add_argument("--bench-warmup", type=int, default=25)
    p.add_argument("--bench-rep", type=int, default=80)
    p.add_argument("--no-grad-ckpt", action="store_true", help="Disable per-block activation checkpointing.")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Spawn and print dashboard URL (same caveat as modal_leaderboard_benchmark: use with care).",
    )
    p.add_argument(
        "-w",
        "--write-result",
        metavar="PATH",
        help="If --no-wait is off, write the JSON string return value to this file.",
    )
    args = p.parse_args()

    fn = modal.Function.from_name(LEADERBOARD_MODAL_APP_NAME, LEADERBOARD_MODAL_REMOTE_FUNCTION_NAME)
    remote_kwargs = dict(
        bench_warmup=args.bench_warmup,
        bench_rep=args.bench_rep,
        grad_ckpt=not args.no_grad_ckpt,
    )

    if args.no_wait:
        call = fn.spawn(**remote_kwargs)
        print("Leaderboard benchmark enqueued (deployed app).", flush=True)
        print(call.get_dashboard_url(), flush=True)
        print(f"function_call_id={call.object_id}", flush=True)
        return

    out = fn.remote(**remote_kwargs)
    print(out)
    if args.write_result:
        Path(args.write_result).write_text(out + "\n")


if __name__ == "__main__":
    main()
