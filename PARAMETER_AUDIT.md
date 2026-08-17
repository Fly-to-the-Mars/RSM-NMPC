# Parameter Audit

This file records the main parameter checks against the manuscript simulation
section. It is intended to make the release workspace easier to audit without
mixing paper table statistics with representative ROS logs.

## Core Parameters

| Manuscript quantity | Manuscript value | Release location |
| --- | ---: | --- |
| Sampling period `T_s` | 0.10--0.15 s | `rsm_sim/config.py`, ROS `nmpc_dt` launch args |
| Horizon `N` | 8 | `ControllerConfig.horizon` |
| Barrier rates `(gamma1, gamma2)` | (0.30, 0.35) | `ControllerConfig.gamma1/gamma2` |
| Support samples `N_b` | 10--16 | `body_constraint_samples` in Python/ROS configs |
| Superellipse axes `(a_x,a_y)` | (0.25, 0.09) m | `body_axes` |
| Minimum clearance `d_min` | 0.08 m | `ControllerConfig.d_min`, ROS launch args |
| Support margin `rho_b` | 0.008 m | `ControllerConfig.rho_b`, ROS launch args |
| LSE parameters `(lambda, lambda_R)` | (10, 35) m^-1 | `lse_lambda`, `rsm_lambda` |
| Confidence inflation | 0.040--0.045 m | `confidence_inflation` |
| Horizon tightening growth | 0.003--0.004 m/step | `horizon_inflation_growth` |
| Velocity/acceleration limits | 2.8--5.0 m/s, 4.2--8.0 m/s^2 | Python defaults and ROS launch args |

## Reproduction Data

- `01_dense_obstacle_agile_flight/data/` contains representative ROS closed-loop
  logs. `paper_dense_obstacle_table.csv` contains the manuscript 60-trial table.
- `02_composite_clutter_arena/data/` contains the dynamic-obstacle figure data
  synchronized with the current figure generation script.
- `03_parameter_sensitivity/paper_sensitivity_table.csv` contains the exact
  manuscript sensitivity table. The generated simulator summaries can vary with
  solver timing and random seeds.

## ROS Runtime Note

The ROS launch stack exposes `rsm_gamma1` and `rsm_gamma2` as implementation
overrides for the RSM controller. These values affect closed-loop behavior
directly and are intentionally kept as launch-level tuning parameters for the
live RViz demonstrations. The paper-level default barrier rates remain encoded
in `ControllerConfig.gamma1/gamma2`.
