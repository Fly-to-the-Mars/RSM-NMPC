#!/usr/bin/env python3
"""Run the Parameter Sensitivity experiment from the lightweight NMPC simulator."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsm_sim.config import SimConfig  # noqa: E402
from rsm_sim.run_experiments import print_summary_table, run_suite  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--scenario", default="dense", choices=["dense", "paper_dense_forest"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = SimConfig(max_steps=args.max_steps) if args.max_steps is not None else SimConfig()
    _, summary = run_suite("sensitivity", [args.scenario], args.trials, sim, args.out)
    print_summary_table(summary)
    print(f"Wrote {args.out / 'sensitivity_summary.csv'}")


if __name__ == "__main__":
    main()
