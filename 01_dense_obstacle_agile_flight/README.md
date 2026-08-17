# 01 Dense-Obstacle Agile Flight

Paper subsection: `Dense-Obstacle Agile Flight`.

## Direct Figure Reproduction

This uses the saved ROS CSV logs in `data/`:

```bash
python make_figure.py
```

Outputs:

- `outputs/sim_dense_density_sweep.pdf`
- `outputs/sim_dense_density_sweep.png`
- `outputs/super_density_ros_summary.csv`
- `outputs/super_density_ros_trials.csv`

## Full ROS/SUPER Re-Run

The ROS assets under `ros_assets/` preserve the launch and generation scripts
used for the density sweep. From this experiment directory:

```bash
SUPER_WS=/path/to/super_ws \
OUT_DIR=$PWD/outputs/ros_logs \
TRIAL_SECONDS=35 \
RVIZ=false \
ros_assets/scripts/run_super_density_sweep.sh
```

Then regenerate the figure:

```bash
python make_figure.py
```

The plotting script expects files named
`super_density_{d1,d2,d3,d4}_{nominal,nmpc_dc,dhocbf_fixed,proposed}.csv`.

## Manuscript Table Data

`paper_dense_obstacle_table.csv` contains the manuscript-level table values for
the dense-obstacle experiment. In particular, the SR values in that table are
computed over 60 randomized trials per density. The saved ROS logs in `data/`
are deterministic representative closed-loop runs used for trajectory and
figure reproduction, so their raw log summary can report SR=100 even when the
paper table reports density-dependent success rates.
