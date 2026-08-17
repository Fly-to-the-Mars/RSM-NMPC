# RSM-NMPC

Simulation-validation and ROS1 closed-loop reproduction code for:

**Recoverable Safety-Margin NMPC for Safety-Critical Navigation of
Euler-Lagrange Robots in Cluttered Environments**

This repository contains the code needed to reproduce the simulation-validation
experiments and to run the live RViz demonstrations. It intentionally contains
only the simulation/reproduction workspace. Paper-only figure post-processing
files and manuscript source files are not included.

## What Is Included

```text
RSM-NMPC/
├── README.md
├── requirements.txt
├── Makefile
├── check_workspace.py
├── run_all.py
├── setup_env.sh
├── rsm_sim/                         # core Python simulator and NMPC modules
├── sim_validation/                  # compatibility import path used by ROS nodes
├── 01_dense_obstacle_agile_flight/  # dense-obstacle experiment reproduction
├── 02_composite_clutter_arena/      # composite arena, ablations, success rates
├── 03_parameter_sensitivity/        # parameter sensitivity scripts and tables
└── ros1_ws/                         # ROS Noetic source workspace for RViz demos
```

The ROS workspace is source-only. Generated `build/`, `devel/`, and log
directories are not committed.

## Tested Environment

Python-only reproduction:

- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `casadi`

Live ROS/RViz reproduction:

- Ubuntu 20.04
- ROS Noetic
- RViz
- Catkin
- PCL, OpenCV, Eigen, GLFW/GLEW, YAML-CPP

## Quick Start: Python-Only Reproduction

This path does not require ROS. It regenerates the main simulation outputs from
saved CSV/PCD assets.

```bash
git clone git@github.com:Fly-to-the-Mars/RSM-NMPC.git
cd RSM-NMPC
python3 -m pip install -r requirements.txt
```

Check that required files and Python dependencies are available:

```bash
python3 check_workspace.py
```

Expected output:

```text
Workspace check passed.
```

Regenerate the two main simulation figures:

```bash
python3 run_all.py --skip-sensitivity
```

Generated outputs:

```text
01_dense_obstacle_agile_flight/outputs/sim_dense_density_sweep.pdf
02_composite_clutter_arena/outputs/sim_composite_arena_rsm.pdf
02_composite_clutter_arena/outputs/sim_composite_arena_success_rate.pdf
```

Run the default reproduction, including a short parameter-sensitivity smoke
test:

```bash
python3 run_all.py
```

Run the full parameter-sensitivity reproduction:

```bash
python3 run_all.py --full-sensitivity
```

The full sensitivity run is slower because it repeats all parameter settings.

## Experiment 1: Dense-Obstacle Agile Flight

This experiment reproduces the dense SUPER-style obstacle density sweep.

```bash
python3 01_dense_obstacle_agile_flight/make_figure.py
```

Output:

```text
01_dense_obstacle_agile_flight/outputs/sim_dense_density_sweep.pdf
```

The script uses saved ROS logs and D1-D4 density map assets. The four densities
correspond to the dense-obstacle results reported in the paper.

## Experiment 2: Composite Clutter Arena and Certificate-Chain Ablations

This experiment reproduces the composite arena figure, including:

- composite arena trajectory diagnostics
- certificate chain `c_k`, `Delta_k`, and `h_R`
- speed-recoverability trend
- SOTA-style offline comparison
- certificate-chain ablations
- speed stress test
- dynamic-obstacle success-rate panel

```bash
python3 02_composite_clutter_arena/make_figure.py
```

Outputs:

```text
02_composite_clutter_arena/outputs/sim_composite_arena_rsm.pdf
02_composite_clutter_arena/outputs/sim_composite_arena_success_rate.pdf
```

The dynamic-obstacle success-rate panel is generated from trial-level data. The
paper setting is:

```text
LDHOCBF: 9/20
EGO: 3/20
SUPER: 8/20
RSM-NMPC: 19/20
```

To update the success-rate panel with newly generated trial data, replace:

```text
02_composite_clutter_arena/data/composite_arena_success_trials.csv
```

and rerun:

```bash
python3 02_composite_clutter_arena/make_figure.py
```

## Experiment 3: Parameter Sensitivity

Smoke test:

```bash
python3 03_parameter_sensitivity/run_parameter_sensitivity.py --trials 2 --max-steps 70
```

Full run:

```bash
python3 03_parameter_sensitivity/run_parameter_sensitivity.py --trials 20
```

Outputs:

```text
03_parameter_sensitivity/outputs/sensitivity_summary.csv
03_parameter_sensitivity/outputs/sensitivity_trials.csv
03_parameter_sensitivity/outputs/sensitivity_dense_*_trajectories.pdf
```

## Optional Virtual Environment

```bash
bash setup_env.sh
source .venv/bin/activate
python3 run_all.py --skip-sensitivity
```

## Makefile Shortcuts

```bash
make check              # check assets and Python dependencies
make figures            # regenerate the two main simulation figures
make reproduce          # check + figures + sensitivity smoke test
make sensitivity        # short sensitivity smoke test
make sensitivity-full   # full sensitivity run
make ros-build          # build the ROS workspace
make ros-density        # launch dense-obstacle RViz demo
make ros-arena          # launch composite-arena RViz demo
make clean              # remove Python caches and ROS build products
```

## ROS1 Closed-Loop RViz Simulation

The live ROS path launches:

- MARSIM / perfect-drone simulation
- RSM-NMPC ROS node
- local point-cloud perception
- predicted NMPC horizon
- executed trajectory
- embodied safety envelope and support points
- nearest-clearance markers
- RSM certificate overlays
- RViz visualization

### Install ROS Dependencies

On Ubuntu 20.04 with ROS Noetic:

```bash
sudo apt update
sudo apt install ros-noetic-desktop-full \
  ros-noetic-pcl-ros ros-noetic-rosfmt \
  libeigen3-dev libpcl-dev libopencv-dev \
  libglfw3-dev libglew-dev libyaml-cpp-dev
```

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### Build the ROS Workspace

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash
./build_ros_workspace.sh
source devel/setup.bash
```

The build script runs:

```bash
catkin_make -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

If your OpenCV CMake path is different:

```bash
OpenCV_DIR=/path/to/opencv4/cmake ./build_ros_workspace.sh
```

## Dense-Obstacle RViz Demos

Run the proposed controller in each density level:

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash

./run_density_rviz.sh d1 proposed
./run_density_rviz.sh d2 proposed
./run_density_rviz.sh d3 proposed
./run_density_rviz.sh d4 proposed
```

Arguments:

- density: `d1`, `d2`, `d3`, or `d4`
- controller: `nominal`, `nmpc_dc`, `dhocbf_fixed`, or `proposed`

Baseline examples:

```bash
./run_density_rviz.sh d3 nominal
./run_density_rviz.sh d3 nmpc_dc
./run_density_rviz.sh d3 dhocbf_fixed
./run_density_rviz.sh d3 proposed
```

The wrapper writes CSV logs to:

```text
01_dense_obstacle_agile_flight/outputs/ros_live/
```

Direct launch command:

```bash
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch \
  rviz:=true \
  density:=d3 \
  controller:=proposed \
  log_csv:=$PWD/../01_dense_obstacle_agile_flight/outputs/ros_live/d3_proposed.csv
```

Run all four density levels sequentially:

```bash
CONTROLLER=proposed RVIZ=true ./run_density_sweep_rviz.sh
```

Close RViz / roslaunch after each density level to continue to the next one.

## Composite Arena RViz Demo

Run the proposed controller:

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash

./run_rescue_arena_rviz.sh proposed
```

Run baseline variants:

```bash
./run_rescue_arena_rviz.sh dhocbf_fixed
./run_rescue_arena_rviz.sh proposed
```

The wrapper writes CSV logs to:

```text
02_composite_clutter_arena/outputs/ros_live/
```

Direct launch command:

```bash
roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch \
  rviz:=true \
  controller:=proposed \
  log_csv:=$PWD/../02_composite_clutter_arena/outputs/ros_live/rescue_arena_proposed.csv
```

Useful visualization arguments:

```bash
roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch \
  rviz:=true \
  controller:=proposed \
  semantic_markers:=true \
  show_perception_markers:=true \
  show_prediction_tube:=true \
  show_certificate_bars:=true \
  show_nearest_clearance:=true
```

## Headless ROS Runs

If no graphical display is available, disable RViz:

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash
source devel/setup.bash

RVIZ=false ./run_density_rviz.sh d3 proposed
RVIZ=false ./run_rescue_arena_rviz.sh proposed
```

Timed terminal-only check:

```bash
timeout --foreground -s INT 45s bash -lc \
  'source /opt/ros/noetic/setup.bash && source devel/setup.bash && RVIZ=false ./run_density_rviz.sh d3 proposed'
```

After a timed run, check that a CSV log was written:

```bash
ls -lh ../01_dense_obstacle_agile_flight/outputs/ros_live/
```

## Regenerate Dense-Obstacle Figure from New ROS Logs

After running live density sweeps, regenerate the dense-obstacle figure from
the new logs:

```bash
cd ..
python3 01_dense_obstacle_agile_flight/super_density_ros_analysis.py \
  --log-dir 01_dense_obstacle_agile_flight/outputs/ros_live \
  --out 01_dense_obstacle_agile_flight/outputs
```

Output:

```text
01_dense_obstacle_agile_flight/outputs/sim_dense_density_sweep.pdf
```

## Output Locations

Python reproduction outputs:

```text
01_dense_obstacle_agile_flight/outputs/
02_composite_clutter_arena/outputs/
03_parameter_sensitivity/outputs/
```

Live ROS logs:

```text
01_dense_obstacle_agile_flight/outputs/ros_live/
02_composite_clutter_arena/outputs/ros_live/
```

## Troubleshooting

### `Missing devel/setup.bash`

Build the ROS workspace first:

```bash
cd ros1_ws
source /opt/ros/noetic/setup.bash
./build_ros_workspace.sh
source devel/setup.bash
```

### `roslaunch: command not found`

Source ROS Noetic:

```bash
source /opt/ros/noetic/setup.bash
```

If ROS Noetic is not installed, install `ros-noetic-desktop-full`.

### RViz does not open

Check that a graphical display is available:

```bash
echo $DISPLAY
```

For SSH sessions, use X11 forwarding or run headless with `RVIZ=false`.

### ROS log directory warning

ROS may warn that `~/.ros/log` is large. This is an environment warning, not a
package error. It can be cleaned with:

```bash
rosclean purge
```

### OpenCV CMake path error

Pass the correct OpenCV path:

```bash
OpenCV_DIR=/path/to/opencv4/cmake ./build_ros_workspace.sh
```

## Validation

Before release, the workspace was checked with:

```bash
python3 check_workspace.py
python3 run_all.py --skip-check --skip-sensitivity
python3 -m py_compile $(find rsm_sim sim_validation 01_dense_obstacle_agile_flight 02_composite_clutter_arena 03_parameter_sensitivity -name '*.py' -print)
```

The ROS workspace is source-only and should be built locally by reviewers using
the commands above.
