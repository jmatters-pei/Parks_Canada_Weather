"""Run the full 6-step pipeline (01 -> 06) sequentially.

Usage:
    python src/00_run_all.py

Notes:
- Uses the current interpreter (sys.executable) to run each step.
- Runs from the repository root so relative paths in step scripts resolve.
- Stops immediately if a step fails and returns that step's exit code.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_step(step_path: Path) -> None:
    root = _repo_root()
    rel = step_path.relative_to(root)
    print("\n" + "=" * 80, flush=True)
    print(f"Running {rel}", flush=True)
    print("=" * 80, flush=True)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")

    subprocess.run(
        [sys.executable, str(step_path)],
        cwd=str(root),
        env=env,
        check=True,
    )


def main() -> int:
    root = _repo_root()

    steps = [
        root / "src" / "01_obtain.py",
        root / "src" / "02_scrub.py",
        root / "src" / "03_explore.py",
        root / "src" / "04_model_fwi.py",
        root / "src" / "05_model_redundancy.py",
        root / "src" / "06_interpret.py",
    ]

    missing = [p for p in steps if not p.exists()]
    if missing:
        print("Missing step script(s):")
        for p in missing:
            print(f"- {p}")
        return 2

    try:
        for step in steps:
            run_step(step)
    except subprocess.CalledProcessError as exc:
        print("\nPipeline failed.")
        print(f"Command: {exc.cmd}")
        print(f"Exit code: {exc.returncode}")
        return int(exc.returncode) if exc.returncode is not None else 1

    print("\nPipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
