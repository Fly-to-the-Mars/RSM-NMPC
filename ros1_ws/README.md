# ROS1 Closed-Loop Simulation Workspace

This workspace contains the paper-facing ROS closed-loop simulation source for
real-time `roslaunch` and RViz visualization. It is the reviewer-facing runtime
workspace for obstacle-condition flight demonstrations.

```text
ros1_ws/src/
├── tv_dhocbf_super
└── SUPER/mars_uav_sim/
    ├── mars_quadrotor_msgs
    ├── marsim_render
    └── perfect_drone_sim
```

The root Python-only reproduction path remains useful for exact figure/table
regeneration from saved logs. This ROS workspace is for live closed-loop
simulation: RViz, onboard local point clouds, the executed trajectory, the
embodied safety envelope, support points, and RSM certificate overlays.

## External Dependencies

The workspace assumes Ubuntu/ROS Noetic and the usual MARSIM rendering
dependencies:

```bash
sudo apt install ros-noetic-desktop-full \
  ros-noetic-pcl-ros ros-noetic-rosfmt \
  libeigen3-dev libpcl-dev libopencv-dev \
  libglfw3-dev libglew-dev libyaml-cpp-dev
```

## Build

```bash
cd ros1_ws
python3 -m pip install -r ../requirements.txt
./build_ros_workspace.sh
```

## Live RViz Launches

Dense-obstacle flight under the four paper density levels:

```bash
cd ros1_ws
source devel/setup.bash

./run_density_rviz.sh d1 proposed
./run_density_rviz.sh d2 proposed
./run_density_rviz.sh d3 proposed
./run_density_rviz.sh d4 proposed
```

Composite rescue/clutter arena:

```bash
./run_rescue_arena_rviz.sh proposed
```

Controller variants:

```bash
./run_density_rviz.sh d3 nominal
./run_density_rviz.sh d3 nmpc_dc
./run_density_rviz.sh d3 dhocbf_fixed
./run_density_rviz.sh d3 proposed
```

For non-visual metric sweeps:

```bash
cd ..
OUT_DIR=$PWD/01_dense_obstacle_agile_flight/outputs/ros_logs \
TRIAL_SECONDS=35 RVIZ=false \
ros1_ws/src/tv_dhocbf_super/scripts/run_super_density_sweep.sh
```

Then regenerate the paper-style dense-obstacle figure from the new logs:

```bash
python 01_dense_obstacle_agile_flight/super_density_ros_analysis.py \
  --log-dir 01_dense_obstacle_agile_flight/outputs/ros_logs \
  --out 01_dense_obstacle_agile_flight/outputs
```
