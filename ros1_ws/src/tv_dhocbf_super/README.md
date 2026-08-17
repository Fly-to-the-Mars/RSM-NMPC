# TV-DHOCBF / RSM-NMPC ROS Simulation Stack

This repository is organized around one main ROS package:
`super_ws/src/tv_dhocbf_super`. It contains the paper-facing launch files,
ROS wrappers, visualization, logging, and the bridge from external simulators
to the RSM-NMPC controller.

## Workspace Layout

```text
RSM_nmpc/simulation_code/
├── rsm_sim/                   # controller, geometry, and batch simulation code
├── sim_validation/            # compatibility import path used by ROS nodes
└── ros1_ws/                   # ROS1 workspace for RViz/SUPER-style demos
    └── src/
        ├── tv_dhocbf_super/   # launches, ROS node, markers, logging
        └── SUPER/             # minimal MARSIM/perfect-drone simulator assets
```

Use `ros1_ws` as the runtime workspace. The canonical launch command comes
from `tv_dhocbf_super`.

## Build

Build SUPER and the main package:

```bash
cd RSM_nmpc/simulation_code
source /opt/ros/noetic/setup.bash
cd ros1_ws
catkin_make -DOpenCV_DIR=/usr/lib/x86_64-linux-gnu/cmake/opencv4
```

For normal use, source only `super_ws`:

```bash
cd RSM_nmpc/simulation_code
source /opt/ros/noetic/setup.bash
source ros1_ws/devel/setup.bash
```

## Canonical Launches

All main scenes are now launched through `roslaunch tv_dhocbf_super ...`.

### 1. Cylinder Demo

Lightweight RViz-only benchmark for quick debugging of the controller, embodied
envelope, local perception model, clearance, and barrier values.

```bash
roslaunch tv_dhocbf_super tvdhocbf_cylinder_demo.launch rviz:=true controller:=proposed
```

Useful variants:

```bash
roslaunch tv_dhocbf_super tvdhocbf_cylinder_demo.launch scenario:=dense controller:=proposed
roslaunch tv_dhocbf_super tvdhocbf_cylinder_demo.launch scenario:=narrow controller:=dhocbf_fixed
roslaunch tv_dhocbf_super tvdhocbf_cylinder_demo.launch mode:=compare
```

This scene uses the early `tv_dhocbf_rviz` simulator, but the canonical entry
is now the launch file above.

### 2. Dense Obstacle / Forest Benchmarks

Paper-facing dense-obstacle agile-flight scene. For the first TRO simulation
validation scene, use SUPER's original dense random-map geometry.  The D1--D4
maps are deterministic density variants segmented from
`random_map_2_26609.pcd`.  The launch still uses our `yopo_forest.rviz`
layout: `/global_pc` is republished for visualization as
`/tvdhocbf_visual/global_pc_blue`, whose obstacle rods carry an explicit
non-rainbow light-blue-to-medium-blue-to-deep-blue RGB gradient. The online perceived
cloud `/cloud_registered` stays low-alpha yellow so it is easy to distinguish
the local `d_lower` input from the static map without washing the rods red.

```bash
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=proposed
```

Four paper density levels:

```bash
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d1 controller:=proposed
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d2 controller:=proposed
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=proposed
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d4 controller:=proposed
```

Later controller comparison on the same scene:

```bash
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=nominal
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=nmpc_dc
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=dhocbf_fixed
roslaunch tv_dhocbf_super tv_dhocbf_super_density.launch rviz:=true density:=d3 controller:=proposed
```

The YOPO-style deterministic forest remains available for GUI debugging and
perception-heavy demos:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch rviz:=true controller:=proposed
```

Change forest density:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch density:=d1 controller:=proposed
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch density:=d2 controller:=proposed
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch density:=d3 controller:=proposed
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch density:=d4 controller:=proposed
```

The current D1--D4 maps contain 61, 65, 105, and 135 trees respectively. The
default backend is `backend:=pcd`; use `backend:=native tree_dist:=...` only for
interactive YOPO-native visual exploration.

Batch dense-obstacle comparison and paper figure generation:

```bash
OUT_DIR=$PWD/sim_validation/results/super_density_ros \
TRIAL_SECONDS=35 RVIZ=false \
super_ws/src/tv_dhocbf_super/scripts/run_super_density_sweep.sh

python3 sim_validation/super_density_ros_analysis.py \
  --log-dir sim_validation/results/super_density_ros \
  --out sim_validation/results/super_density_ros \
  --log-prefix super_density \
  --pcd-prefix tvdhocbf_super_density \
  --figure-name fig_super_density_sweep.pdf
```

For the native backend, the YOPO bridge maps:

- `/sim/odom` -> `/lidar_slam/odom`
- `/lidar_points` -> `/cloud_registered`
- `/mock_map` -> `/global_pc`

For the default PCD backend, MARSIM/perfect-drone publishes `/lidar_slam/odom`,
`/cloud_registered`, and `/global_pc` directly. In both cases, the local
RSM-NMPC safety constraints use the online local cloud, while the global cloud
is latched and downsampled for guide-path extraction and RViz rendering.

### 3. Complex Rescue Arena

Custom deterministic scenario designed to show the paper's main advantages:
embodied safety envelope, recoverable safety margin, local safety-set updates,
and agile safe traversal.

```bash
roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch rviz:=true controller:=proposed
```

Optional labels:

```bash
roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch semantic_labels:=true controller:=proposed
```

This launch wraps `tvdhocbf_rescue_arena.launch`, which loads the custom PCD
map and semantic RViz markers.

## Controller Switches

Use the same scene and switch only the controller:

```bash
controller:=nominal
controller:=nmpc_dc
controller:=dhocbf_fixed
controller:=proposed
controller:=fixed_sphere
controller:=static_dhocbf
controller:=pointwise_gp
```

Example:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch controller:=dhocbf_fixed tree_dist:=4.6
roslaunch tv_dhocbf_super tvdhocbf_complex_arena.launch controller:=static_dhocbf
```

## RViz Safety Overlays

The default view is `visual_detail:=clean`, which keeps the paper-facing
signals but hides dense debugging geometry:

- Blue-gradient point cloud: obstacle rods remain in a clean SUPER-style
  light-blue/medium-blue/deep-blue palette for guide-path extraction and RViz context.
- Yellow local points: current LiDAR slice entering `d_lower`.
- Faint blue ring: local sensing horizon.
- Green body footprint: attitude-dependent full-body envelope `B(q)`.
- Red support points: sampled full-body support set `S(q)`.
- Cyan ring: static safety padding; orange/red-orange ring: recovery reserve
  and RSM tension. Its radius is the speed-dependent reserve, visually
  amplified by `reserve_visual_gain`; its color and thickness react to
  `Delta` and `h_R`, so it becomes redder/thicker when the recoverable
  certificate is tight near obstacles. The controller and CSV still use the
  true metric values.
- Red/orange/green segment: active nearest full-body clearance witness.
- Thin blue line: predicted NMPC center trajectory.
- Three small bars near the vehicle: `c_body`, `Delta`, and `h_R`.

For debugging the full certificate construction, switch to `visual_detail:=debug`.
This restores the clutter-heavy layers: every local obstacle inflation disk,
sensor rays, support rays, and predicted recovery tubes.

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch \
  controller:=proposed visual_detail:=debug
```

For an even cleaner recording view:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch \
  controller:=proposed show_prediction_tube:=false show_certificate_bars:=false
```

If the dynamic reserve is still hard to see from a far top-down camera, increase
only the visualization gain:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch \
  controller:=proposed reserve_visual_gain:=6.0
```

## Logging

Add `log_csv:=...` to any main launch:

```bash
roslaunch tv_dhocbf_super tvdhocbf_dense_forest.launch \
  controller:=proposed density:=d3 log_csv:=/tmp/tvdhocbf_dense_proposed.csv
```

The CSV records state, reference, command, feasibility, solve time, full-body
clearance, barrier value, recoverable margin, dynamic reserve, speed, local
obstacle count, and guide-path size.

The proposed controller logs the full RSM-NMPC certificate chain:

```text
d_lower_min, c_body, gamma_ref, recovery_energy, recoverability_delta, h_R,
pred_min_d_lower, pred_min_c_body, pred_min_delta, pred_min_h_R,
pred_min_psi0_constraint, pred_min_psi1_constraint,
pred_min_psi2_constraint, max_safety_slack
```

Check a log with:

```bash
python3 sim_validation/check_rsm_certificate.py /tmp/tvdhocbf_dense_proposed.csv
```

For a valid RSM-NMPC run, the diagnostic minima of `c_body`, `Delta`
(`recoverability_delta`), `h_R`, and the sampled-data recursion margins should
be nonnegative up to numerical tolerance. `max_safety_slack` is reported
separately so softening cannot hide certificate violations.

## Core Algorithm Files

The implementation is intentionally split between reusable Python modules and
ROS wrappers:

```text
sim_validation/config.py
  Controller parameters, baselines, ablations, RSM/NMPC weights.

sim_validation/geometry.py
  d_lower, embodied support points S(q), smooth full-body clearance c_k,
  recoverability Delta_k, h_R, and barrier-recursion diagnostics.

sim_validation/nmpc.py
  CasADi receding-horizon OCP enforcing h_R through sampled-data recursion.

super_ws/src/tv_dhocbf_super/scripts/tv_dhocbf_super_node.py
  ROS node that receives odometry/point clouds, extracts local obstacles,
  plans a guide path, solves RSM-NMPC, publishes commands and visualization.

super_ws/src/tv_dhocbf_super/scripts/yopo_topic_bridge.py
  Converts YOPO native simulator topics to the SUPER/RSM-NMPC topic contract.

super_ws/src/tv_dhocbf_super/scripts/yopo_native_drone.py
  Lightweight low-level executor used by the YOPO native forest scene.
```

## Main ROS Interfaces

Inputs used by `tv_dhocbf_super_node.py`:

```text
/lidar_slam/odom
/cloud_registered
/global_pc
/planning/click_goal
/goal
/move_base_simple/goal
```

Outputs:

```text
/planning/pos_cmd
/fsm_node/fsm/path
/fsm_node/visualization/exp_traj
/fsm_node/visualization/points
/tv_dhocbf_super/metrics
```

Do not launch SUPER's native `fsm_node` together with this package, because
both publish `/planning/pos_cmd`.

## Legacy And Auxiliary Launches

The following launches are kept for earlier experiments and batch scripts:

```text
tv_dhocbf_click.launch              # base SUPER/MARSIM wrapper
tv_dhocbf_dense.launch              # old dense PCD scene
tv_dhocbf_high_speed.launch         # old high-speed PCD scene
tv_dhocbf_super_density.launch      # SUPER-style density sweep
tvdhocbf_rsm_clutter_corridor.launch # compatibility alias for SUPER density
tvdhocbf_yopo_native_forest.launch  # implementation behind dense forest
tvdhocbf_rescue_arena.launch        # implementation behind complex arena
tvdhocbf_yopo_forest.launch         # generated YOPO-style PCD forest
tvdhocbf_agile_forest.launch        # older structured forest scene
```

For paper-facing runs, prefer the three canonical launches:

```text
tvdhocbf_cylinder_demo.launch
tv_dhocbf_super_density.launch
tvdhocbf_complex_arena.launch
```
