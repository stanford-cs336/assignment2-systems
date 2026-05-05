from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Checkout root (directory that contains ``cs336_systems/``). Falls back to Modal ``/workspace``."""
    here = Path(__file__).resolve()
    for root in list(here.parents) + [Path("/workspace")]:
        if (root / "cs336_systems").is_dir():
            return root
    raise RuntimeError("Cannot locate directory cs336_systems/ (repo root).")
