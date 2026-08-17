#!/usr/bin/env python3
"""Check that the simulation-validation workspace is complete."""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parent


REQUIRED_PATHS = [
    ROOT / "README.md",
    ROOT / "PARAMETER_AUDIT.md",
    ROOT / "requirements.txt",
    ROOT / "EXPERIMENT_INDEX.md",
    ROOT / "rsm_sim" / "nmpc.py",
    ROOT / "01_dense_obstacle_agile_flight" / "make_figure.py",
    ROOT / "01_dense_obstacle_agile_flight" / "super_density_ros_analysis.py",
    ROOT / "01_dense_obstacle_agile_flight" / "paper_dense_obstacle_table.csv",
    ROOT / "02_composite_clutter_arena" / "make_figure.py",
    ROOT / "02_composite_clutter_arena" / "composite_arena_validation.py",
    ROOT / "03_parameter_sensitivity" / "run_parameter_sensitivity.py",
    ROOT / "03_parameter_sensitivity" / "paper_sensitivity_table.csv",
    ROOT / "sim_validation" / "nmpc.py",
    ROOT / "ros1_ws" / "src" / "tv_dhocbf_super" / "package.xml",
    ROOT / "ros1_ws" / "src" / "SUPER" / "mars_uav_sim" / "mars_quadrotor_msgs" / "package.xml",
    ROOT / "ros1_ws" / "src" / "SUPER" / "mars_uav_sim" / "marsim_render" / "package.xml",
    ROOT / "ros1_ws" / "src" / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "package.xml",
    ROOT / "ros1_ws" / "build_ros_workspace.sh",
    ROOT / "ros1_ws" / "run_density_rviz.sh",
    ROOT / "ros1_ws" / "run_rescue_arena_rviz.sh",
]

REQUIRED_PACKAGES = ["numpy", "pandas", "matplotlib", "scipy", "casadi"]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict-ros",
        action="store_true",
        help="also require roslaunch and catkin_make to be visible on PATH",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def check_paper_tables(errors: list[str]) -> None:
    dense_table = ROOT / "01_dense_obstacle_agile_flight" / "paper_dense_obstacle_table.csv"
    sensitivity_table = ROOT / "03_parameter_sensitivity" / "paper_sensitivity_table.csv"
    composite_success = ROOT / "02_composite_clutter_arena" / "data" / "composite_arena_success_summary.csv"

    if dense_table.exists():
        rows = read_csv_rows(dense_table)
        rsm_aggregate = [
            r for r in rows if r.get("density") == "Aggregate" and r.get("controller") == "RSM-NMPC"
        ]
        if not rsm_aggregate or abs(float(rsm_aggregate[0].get("SR_%", "nan")) - 99.6) > 1e-6:
            errors.append("paper dense-obstacle table does not contain Aggregate/RSM-NMPC SR=99.6")

    if sensitivity_table.exists():
        rows = read_csv_rows(sensitivity_table)
        lambda_values = {r.get("parameter_value") for r in rows if r.get("sensitivity_group") == "lambda"}
        if lambda_values != {"lambda=5", "lambda=10", "lambda=20"}:
            errors.append("paper sensitivity table lambda group must be lambda=5, lambda=10, lambda=20")

    if composite_success.exists():
        rows = read_csv_rows(composite_success)
        expected = {
            "dhocbf_fixed": 9,
            "super_style": 3,
            "ego_style": 8,
            "proposed": 19,
        }
        observed = {r.get("method"): int(float(r.get("success_count", "nan"))) for r in rows}
        for method, count in expected.items():
            if observed.get(method) != count:
                errors.append(f"composite dynamic success count for {method} should be {count}/20")


def main() -> int:
    args = parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for path in REQUIRED_PATHS:
        if not path.exists():
            errors.append(f"missing required path: {path.relative_to(ROOT)}")

    dense_logs = count_files(ROOT / "01_dense_obstacle_agile_flight" / "data", "super_density_*.csv")
    dense_pcd = count_files(
        ROOT / "01_dense_obstacle_agile_flight" / "ros_assets" / "pcd",
        "tvdhocbf_super_density_*.pcd",
    )
    composite_csv = count_files(ROOT / "02_composite_clutter_arena" / "data", "composite_arena_*.csv")
    sensitivity_csv = count_files(ROOT / "03_parameter_sensitivity", "sensitivity_*.csv")
    ros_density_pcd = count_files(
        ROOT / "ros1_ws" / "src" / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd",
        "tvdhocbf_super_density_*.pcd",
    )
    ros_density_yaml = count_files(
        ROOT / "ros1_ws" / "src" / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config",
        "tvdhocbf_super_density_*.yaml",
    )

    if dense_logs < 16:
        errors.append(f"expected at least 16 dense-obstacle CSV logs, found {dense_logs}")
    if dense_pcd < 4:
        errors.append(f"expected four density PCD assets, found {dense_pcd}")
    if composite_csv < 4:
        errors.append(f"expected composite-arena CSV data files, found {composite_csv}")
    if sensitivity_csv < 2:
        errors.append(f"expected sensitivity summary/trials CSVs, found {sensitivity_csv}")
    if ros_density_pcd < 4:
        errors.append(f"expected four ROS density PCD maps, found {ros_density_pcd}")
    if ros_density_yaml < 4:
        errors.append(f"expected four ROS density config YAMLs, found {ros_density_yaml}")

    for pkg in REQUIRED_PACKAGES:
        if not has_module(pkg):
            errors.append(f"missing Python package: {pkg}")

    check_paper_tables(errors)

    roslaunch = shutil.which("roslaunch")
    catkin_make = shutil.which("catkin_make")
    if args.strict_ros:
        if roslaunch is None:
            errors.append("roslaunch not found on PATH")
        if catkin_make is None:
            errors.append("catkin_make not found on PATH")
    else:
        if roslaunch is None or catkin_make is None:
            warnings.append("ROS tools not found; Python-only reproduction is still available")

    if errors:
        print("Workspace check failed:")
        for item in errors:
            print(f"  - {item}")
        if warnings:
            print("Warnings:")
            for item in warnings:
                print(f"  - {item}")
        return 1

    print("Workspace check passed.")
    print(f"  dense logs: {dense_logs}")
    print(f"  density PCD assets: {dense_pcd}")
    print(f"  composite CSV files: {composite_csv}")
    print(f"  sensitivity CSV files: {sensitivity_csv}")
    print(f"  ROS density PCD maps: {ros_density_pcd}")
    print(f"  ROS density YAML configs: {ros_density_yaml}")
    if warnings:
        print("Warnings:")
        for item in warnings:
            print(f"  - {item}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
