# 02 Composite Clutter Arena and Certificate-Chain Ablations

Paper subsection: `Composite Clutter Arena and Certificate-Chain Ablations`.

Run:

```bash
python make_figure.py
```

Outputs:

- `outputs/sim_composite_arena_rsm.pdf`
- `outputs/sim_composite_arena_rsm.png`
- `outputs/sim_composite_arena_success_rate.pdf`
- `outputs/sim_composite_arena_success_rate.png`
- `outputs/composite_arena_summary.csv`
- `outputs/composite_arena_controller_summary.csv`
- `outputs/composite_arena_sota_offline.csv`
- `outputs/composite_arena_ablation.csv`
- `outputs/composite_arena_speed_sweep.csv`
- `outputs/composite_arena_dynamic_obstacles.csv`
- `outputs/composite_arena_success_trials.csv`
- `outputs/composite_arena_success_summary.csv`

The local PCD asset in `assets/` is used only for the arena visualization. The
certificate-chain data are generated deterministically by
`composite_arena_validation.py`.

The scene includes three spherical obstacles that cross the gate,
L-corridor, and slot at fixed piecewise-constant velocities. RSM-NMPC
propagates their positions with a 0.75 s look-ahead and applies lateral,
vertical, and speed responses before evaluating the full-body dynamic
clearance. The EGO-style and SUPER-style traces represent the tested
static-map configurations without an explicit moving-obstacle predictor;
all methods are evaluated against the same realized obstacle trajectories.

The success-rate figure uses 20 shared randomized scenarios per method. A run
is successful only when the goal is reached without static or dynamic
collision. The summary CSV also reports 95% Wilson confidence intervals.
