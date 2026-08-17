#!/usr/bin/env python3
"""Run paper simulation reproductions in manuscript order."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent


def run_step(name: str, cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n== {name} ==", flush=True)
    print("$ " + " ".join(cmd), flush=True)
    started = time.time()
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run(cmd, cwd=str(cwd or ROOT), env=env, check=True)
    elapsed = time.time() - started
    print(f"[done] {name}: {elapsed:.1f} s", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="skip dependency and asset checks before running experiments",
    )
    parser.add_argument(
        "--full-sensitivity",
        action="store_true",
        help="run the table-level sensitivity experiment with 20 trials",
    )
    parser.add_argument(
        "--sensitivity-trials",
        type=int,
        default=None,
        help="override the number of sensitivity trials",
    )
    parser.add_argument(
        "--sensitivity-max-steps",
        type=int,
        default=None,
        help="override sensitivity max steps; default is 70 for smoke mode and unrestricted for full mode",
    )
    parser.add_argument(
        "--skip-sensitivity",
        action="store_true",
        help="only regenerate the two main simulation figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    if not args.skip_check:
        run_step("workspace check", [py, "check_workspace.py"])

    run_step(
        "1. Dense-Obstacle Agile Flight",
        [py, "01_dense_obstacle_agile_flight/make_figure.py"],
    )
    run_step(
        "2. Composite Clutter Arena and Certificate-Chain Ablations",
        [py, "02_composite_clutter_arena/make_figure.py"],
    )

    if not args.skip_sensitivity:
        trials = args.sensitivity_trials
        if trials is None:
            trials = 20 if args.full_sensitivity else 2
        max_steps = args.sensitivity_max_steps
        if max_steps is None and not args.full_sensitivity:
            max_steps = 70
        cmd = [
            py,
            "03_parameter_sensitivity/run_parameter_sensitivity.py",
            "--trials",
            str(trials),
        ]
        if max_steps is not None:
            cmd += ["--max-steps", str(max_steps)]
        run_step("3. Parameter Sensitivity", cmd)

    print("\nAll requested simulation reproductions finished.", flush=True)


if __name__ == "__main__":
    main()
