#!/usr/bin/env python3
"""Run baseline, ablation, and sensitivity simulations.

Examples:
    python3 -m sim_validation.run_experiments --suite ablation --trials 20
    python3 -m sim_validation.run_experiments --suite baseline --scenario dense --trials 5
"""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import SimConfig, ablation_controllers, baseline_controllers, sensitivity_groups
from .nmpc import simulate_controller
from .scenarios import Scenario, build_scenario, plan_grid_path


def attach_guidance(scenario: Scenario, sim: SimConfig) -> Scenario:
    scenario.path = plan_grid_path(scenario, sim.path_resolution, sim.path_margin)
    return scenario


def stable_offset(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 997


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["suite", "scenario", "controller", "controller_label"]
    if "sensitivity_group" in df.columns:
        group_cols.append("sensitivity_group")
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "trials": len(grp),
                "SR_%": 100.0 * grp["success"].mean(),
                "collision_%": 100.0 * grp["collision"].mean(),
                "FR_%": 100.0 * grp["feasibility_rate"].mean(),
                "min_clearance_mean_m": grp["min_clearance_m"].mean(),
                "min_clearance_std_m": grp["min_clearance_m"].std(ddof=0),
                "min_recoverable_margin_mean": grp["min_recoverable_margin"].mean()
                if "min_recoverable_margin" in grp
                else np.nan,
                "avg_speed_mean_mps": grp["avg_speed_mps"].mean(),
                "avg_speed_std_mps": grp["avg_speed_mps"].std(ddof=0),
                "min_active_speed_mean_mps": grp["min_active_speed_mps"].mean(),
                "max_safety_slack_mean": grp["max_safety_slack"].mean() if "max_safety_slack" in grp else np.nan,
                "compute_mean_ms": grp["compute_ms"].mean(),
                "compute_std_ms": grp["compute_ms"].std(ddof=0),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary_table(summary: pd.DataFrame) -> None:
    cols = [
        "suite",
        "scenario",
        "controller_label",
        "SR_%",
        "FR_%",
        "min_clearance_mean_m",
        "min_recoverable_margin_mean",
        "max_safety_slack_mean",
        "avg_speed_mean_mps",
        "compute_mean_ms",
    ]
    existing = [c for c in cols if c in summary.columns]
    with pd.option_context("display.max_rows", 200, "display.width", 160):
        print(summary[existing].sort_values(existing[:3]).to_string(index=False))


def plot_trajectories(
    scenario: Scenario,
    traces: Dict[str, Dict[str, np.ndarray]],
    labels: Dict[str, str],
    out_file: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for obs in scenario.obstacles:
        patch = plt.Circle(obs.center, obs.radius, color="0.25", alpha=0.20)
        ax.add_patch(patch)
        edge = plt.Circle(obs.center, obs.radius, color="0.20", fill=False, linewidth=1.0)
        ax.add_patch(edge)
    if scenario.path is not None:
        ax.plot(scenario.path[:, 0], scenario.path[:, 1], "--", color="0.65", linewidth=1.0, label="coarse guide")
    for key, trace in traces.items():
        xy = trace["states"][:, :2]
        ax.plot(xy[:, 0], xy[:, 1], linewidth=1.8, label=labels.get(key, key))
    ax.scatter([scenario.start[0]], [scenario.start[1]], marker="o", s=45, color="green", zorder=5)
    ax.scatter([scenario.goal[0]], [scenario.goal[1]], marker="*", s=95, color="crimson", zorder=5)
    ax.set_xlim(scenario.bounds[0], scenario.bounds[1])
    ax.set_ylim(scenario.bounds[2], scenario.bounds[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, linewidth=0.35, alpha=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


def run_suite(
    suite: str,
    scenarios: Iterable[str],
    trials: int,
    sim: SimConfig,
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows: List[Dict[str, float]] = []
    first_traces: Dict[Tuple[str, str], Dict[str, Dict[str, np.ndarray]]] = {}
    labels: Dict[str, str] = {}
    scenario_cache: Dict[str, Scenario] = {}

    if suite == "baseline":
        controller_groups = [("baseline", baseline_controllers(sim))]
    elif suite == "ablation":
        controller_groups = [("ablation", ablation_controllers(sim))]
    elif suite == "sensitivity":
        controller_groups = list(sensitivity_groups(sim).items())
    else:
        raise ValueError("Unknown suite: %s" % suite)

    for scenario_name in scenarios:
        for trial in range(trials):
            seed = 1000 + trial
            scenario = attach_guidance(build_scenario(scenario_name, seed), sim)
            if trial == 0:
                scenario_cache[scenario_name] = scenario
            for group_name, controllers in controller_groups:
                for controller_cfg in controllers:
                    metrics, trace = simulate_controller(scenario, controller_cfg, sim, seed + stable_offset(controller_cfg.key))
                    metrics["suite"] = suite
                    metrics["sensitivity_group"] = group_name if suite == "sensitivity" else ""
                    rows.append(metrics)
                    labels[controller_cfg.key] = controller_cfg.label
                    if trial == 0:
                        first_traces.setdefault((scenario_name, group_name), {})[controller_cfg.key] = trace
                    print(
                        "%s/%s trial=%02d controller=%s success=%d min_clearance=%.3f FR=%.2f"
                        % (
                            suite,
                            scenario_name,
                            trial,
                            controller_cfg.key,
                            metrics["success"],
                            metrics["min_clearance_m"],
                            metrics["feasibility_rate"],
                        ),
                        flush=True,
                    )

    raw = pd.DataFrame(rows)
    summary = summarize(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / ("%s_trials.csv" % suite), index=False)
    summary.to_csv(out_dir / ("%s_summary.csv" % suite), index=False)

    for (scenario_name, group_name), trace_group in first_traces.items():
        fig_name = "%s_%s_%s_trajectories.pdf" % (suite, scenario_name, group_name)
        plot_trajectories(scenario_cache[scenario_name], trace_group, labels, out_dir / fig_name)
    return raw, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["baseline", "ablation", "sensitivity", "all"], default="ablation")
    parser.add_argument(
        "--scenario",
        choices=["dense", "narrow", "forest", "paper_dense_forest", "paper_narrow_gap", "super_forest", "ruin", "warehouse", "all"],
        default="dense",
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("sim_validation/results"))
    parser.add_argument("--max-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = SimConfig(max_steps=args.max_steps) if args.max_steps is not None else SimConfig()
    suites = ["baseline", "ablation", "sensitivity"] if args.suite == "all" else [args.suite]
    scenarios = ["dense", "narrow", "forest"] if args.scenario == "all" else [args.scenario]
    all_summaries = []
    for suite in suites:
        suite_scenarios = scenarios
        if suite in {"ablation", "sensitivity"} and args.scenario == "all":
            suite_scenarios = ["dense"]
        _, summary = run_suite(suite, suite_scenarios, args.trials, sim, args.out)
        all_summaries.append(summary)
    merged = pd.concat(all_summaries, ignore_index=True)
    merged.to_csv(args.out / "summary_all.csv", index=False)
    print_summary_table(merged)


if __name__ == "__main__":
    main()
