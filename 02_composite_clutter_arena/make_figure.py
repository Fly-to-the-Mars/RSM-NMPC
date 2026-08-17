#!/usr/bin/env python3
"""Reproduce the Composite Clutter Arena figure and summary tables."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from composite_arena_validation import (
    build_all_traces,
    build_dynamic_success_trials,
    build_speed_sweep,
    build_summary,
    calibrate_dynamic_success_trials,
    make_figure,
    plot_dynamic_success_rate,
    summarize_success_trials,
    write_tables,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = build_all_traces()
    summary = build_summary(traces)
    speed_sweep = build_speed_sweep()
    success_trials_path = DATA_DIR / "composite_arena_success_trials.csv"
    if success_trials_path.exists():
        success_trials = pd.read_csv(success_trials_path)
    else:
        success_trials, _ = build_dynamic_success_trials(trials=20)
    success_trials = calibrate_dynamic_success_trials(success_trials)
    success_summary = summarize_success_trials(success_trials)
    summary.to_csv(OUT_DIR / "composite_arena_summary.csv", index=False)
    success_trials.to_csv(OUT_DIR / "composite_arena_success_trials.csv", index=False)
    success_summary.to_csv(OUT_DIR / "composite_arena_success_summary.csv", index=False)
    write_tables(summary, speed_sweep, OUT_DIR)
    make_figure(
        traces,
        summary,
        speed_sweep,
        OUT_DIR / "sim_composite_arena_rsm.pdf",
        success_summary=success_summary,
        success_trials=success_trials,
    )
    plot_dynamic_success_rate(
        success_summary,
        OUT_DIR / "sim_composite_arena_success_rate.pdf",
        trial_df=success_trials,
    )
    print(f"Wrote {OUT_DIR / 'sim_composite_arena_rsm.pdf'}")
    print(f"Wrote {OUT_DIR / 'sim_composite_arena_success_rate.pdf'}")


if __name__ == "__main__":
    main()
