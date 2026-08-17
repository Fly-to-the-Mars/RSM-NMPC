# Experiment Index

This folder organizes the simulation code in the same order as the
`SIMULATION VALIDATION` section of the RSM-NMPC manuscript.

## 1. Dense-Obstacle Agile Flight

Paper subsection:
`\subsection{Dense-Obstacle Agile Flight}`

Folder:
`01_dense_obstacle_agile_flight/`

Main entry:
`python make_figure.py`

Core files:
- `super_density_ros_analysis.py`: loads the ROS/SUPER density-sweep logs,
  computes the dense-obstacle summary metrics, and regenerates
  `sim_dense_density_sweep.pdf`.
- `data/super_density_*.csv`: saved closed-loop logs for D1--D4 and the
  compared controllers.
- `ros_assets/scripts/run_super_density_sweep.sh`: optional ROS/SUPER helper
  for rerunning the density sweep when the full ROS workspace is available.
- `ros_assets/config/`: controller/map parameter files for D1--D4.
- `ros_assets/pcd/`: deterministic density-map point clouds used by the
  figure script.

Expected outputs:
- `outputs/sim_dense_density_sweep.pdf`
- `outputs/sim_dense_density_sweep.png`
- `outputs/super_density_ros_summary.csv`
- `outputs/super_density_ros_trials.csv`

## 2. Composite Clutter Arena and Certificate-Chain Ablations

Paper subsection:
`\subsection{Composite Clutter Arena and Certificate-Chain Ablations}`

Folder:
`02_composite_clutter_arena/`

Main entry:
`python make_figure.py`

Core files:
- `composite_arena_validation.py`: constructs the composite arena,
  propagates three moving obstacles, generates the controlled baseline and
  static-map planner replay diagnostics, certificate-chain ablations, and
  commanded-speed stress test.
- `assets/tvdhocbf_rescue_arena.pcd`: paper arena point-cloud asset.
- `data/composite_arena_*.csv`: saved figure/table diagnostics.

Expected outputs:
- `outputs/sim_composite_arena_rsm.pdf`
- `outputs/sim_composite_arena_rsm.png`
- `outputs/sim_composite_arena_success_rate.pdf`
- `outputs/sim_composite_arena_success_rate.png`
- `outputs/composite_arena_controller_summary.csv`
- `outputs/composite_arena_sota_offline.csv`
- `outputs/composite_arena_ablation.csv`
- `outputs/composite_arena_speed_sweep.csv`
- `outputs/composite_arena_dynamic_obstacles.csv`
- `outputs/composite_arena_success_trials.csv`
- `outputs/composite_arena_success_summary.csv`
- `outputs/composite_arena_summary.csv`

## 3. Parameter Sensitivity

Paper subsection:
`\subsection{Parameter Sensitivity}`

Folder:
`03_parameter_sensitivity/`

Main entry:
`python run_parameter_sensitivity.py`

Core files:
- `run_parameter_sensitivity.py`: command-line wrapper around the shared
  simulation package for regenerating the parameter-sensitivity CSVs.
- `sensitivity_summary.csv`: reference summary values used by the paper table.
- `sensitivity_trials.csv`: per-trial reference values.

Expected outputs:
- `outputs/sensitivity_summary.csv`
- `outputs/sensitivity_trials.csv`

## Shared RSM-NMPC Simulation Package

Folder:
`rsm_sim/`

Core files:
- `config.py`: simulation/controller parameter dataclasses.
- `geometry.py`: obstacle, clearance, support-envelope, and certificate
  utility functions.
- `nmpc.py`: NMPC/controller implementations and certificate evaluation.
- `scenarios.py`: stochastic and structured scenario builders.
- `run_experiments.py`: shared batch-run helpers used by the sensitivity
  experiment.

## Optional ROS1 Runtime Package

Folder:
`ros1_ws/src/tv_dhocbf_super/`

Purpose:
contains the ROS launch files, visualization helpers, map-generation scripts,
and runtime node used to rerun the SUPER/MARSIM density-sweep simulations. This
workspace also includes the minimal SUPER/MARSIM packages required by the live
RViz launches:
`quadrotor_msgs`, `marsim_render`, and `perfect_drone_sim`. `build/`, `devel/`,
logs, and generated caches are intentionally excluded from the release
workspace.

## Reproducibility Notes

The first two experiments can be regenerated directly from the saved logs and
assets in this release folder. The optional ROS/SUPER rerun path in
`ros1_ws/` still requires a working ROS1 and SUPER/MARSIM workspace. The
parameter-sensitivity experiment is self-contained but can be computationally
expensive when using the full number of trials from the paper.
