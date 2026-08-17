#!/usr/bin/env python3
"""Reproduce the Dense-Obstacle Agile Flight figure from saved ROS logs."""

from __future__ import annotations

from pathlib import Path

from super_density_ros_analysis import aggregate, collect_logs, plot_summary


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "outputs"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = collect_logs(DATA_DIR, "super_density")
    if raw.empty:
        raise RuntimeError(f"No super_density logs found in {DATA_DIR}")

    summary = aggregate(raw)
    raw.to_csv(OUT_DIR / "super_density_ros_trials.csv", index=False)
    summary.to_csv(OUT_DIR / "super_density_ros_summary.csv", index=False)
    plot_summary(
        summary,
        raw,
        OUT_DIR / "sim_dense_density_sweep.pdf",
        "tvdhocbf_super_density",
        "dense obstacle",
    )
    print(f"Wrote {OUT_DIR / 'sim_dense_density_sweep.pdf'}")


if __name__ == "__main__":
    main()
