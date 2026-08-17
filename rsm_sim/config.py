"""Configuration objects for reproducible simulation validation."""

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.15
    max_steps: int = 120
    reach_radius: float = 0.28
    process_noise_acc: float = 0.015
    process_noise_alpha: float = 0.01
    max_obstacles: int = 14
    body_axes: Tuple[float, float] = (0.25, 0.09)
    body_samples: int = 72
    path_resolution: float = 0.12
    path_margin: float = 0.34
    lookahead: float = 1.35


@dataclass(frozen=True)
class ControllerConfig:
    key: str
    label: str
    constraint_mode: str = "rsm"  # none, distance, dhocbf, rsm, rsm_instant
    envelope: str = "superellipse"  # point, sphere, ellipse, superellipse
    horizon: int = 8
    dt: float = 0.15
    d_min: float = 0.08
    rho_b: float = 0.008
    robot_radius: float = 0.25
    body_axes: Tuple[float, float] = (0.25, 0.09)
    body_shape_exponent: float = 4.0
    body_constraint_samples: int = 16
    gamma1: float = 0.30
    gamma2: float = 0.35
    lse_lambda: float = 10.0
    obs_lse_lambda: float = 24.0
    confidence_inflation: float = 0.045
    horizon_inflation_growth: float = 0.004
    delta0: float = 0.002
    delta1: float = 0.001
    delta2: float = 0.0
    use_tightening: bool = True
    rsm_base_margin: float = 0.015
    rsm_tau: float = 0.035
    rsm_brake_gain: float = 0.045
    rsm_terminal_margin: float = 0.010
    rsm_lambda: float = 35.0
    rsm_gamma0: float = 0.380
    rsm_gamma_clearance_gain: float = 0.420
    rsm_gamma_boundary_radius: float = 0.18
    rsm_gamma_boundary_samples: int = 8
    rsm_v_pos: float = 0.020
    rsm_v_vel: float = 0.0035
    rsm_v_theta: float = 0.000
    rsm_v_omega: float = 0.000
    rsm_cost_weight: float = 0.0
    safety_slack_weight: float = 50000.0
    safety_slack_linear: float = 200.0
    safety_slack_max: float = 0.08
    warm_start_blend: float = 0.86
    v_max: float = 2.8
    a_max: float = 4.2
    alpha_max: float = 6.0
    theta_max: float = 1.25
    omega_max: float = 3.0
    q_pos: float = 18.0
    q_vel: float = 0.7
    q_theta: float = 0.02
    r_acc: float = 0.10
    r_alpha: float = 0.04
    r_du: float = 0.018
    terminal_pos: float = 65.0
    ipopt_max_iter: int = 90
    ipopt_max_cpu_time: float = 0.08


def baseline_controllers(sim: SimConfig) -> List[ControllerConfig]:
    base = ControllerConfig(key="proposed", label="RSM-NMPC (proposed)", dt=sim.dt, body_axes=sim.body_axes)
    return [
        replace(
            base,
            key="nominal",
            label="Nominal NMPC",
            constraint_mode="none",
            envelope="point",
            confidence_inflation=0.0,
            horizon_inflation_growth=0.0,
            use_tightening=False,
        ),
        replace(
            base,
            key="nmpc_dc",
            label="NMPC-DC",
            constraint_mode="distance",
            envelope="sphere",
            robot_radius=0.18,
            confidence_inflation=0.015,
            horizon_inflation_growth=0.0,
            use_tightening=False,
        ),
        replace(
            base,
            key="dhocbf_fixed",
            label="NMPC-DHOCBF fixed",
            constraint_mode="dhocbf",
            envelope="sphere",
            robot_radius=0.25,
            confidence_inflation=0.030,
            horizon_inflation_growth=0.0,
            use_tightening=False,
        ),
        base,
    ]


def ablation_controllers(sim: SimConfig) -> List[ControllerConfig]:
    full = ControllerConfig(key="full", label="Full RSM-NMPC", dt=sim.dt, body_axes=sim.body_axes)
    return [
        full,
        replace(
            full,
            key="no_perception_lcb",
            label="- perception lower bound",
            confidence_inflation=0.0,
            horizon_inflation_growth=0.0,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
        ),
        replace(
            full,
            key="fixed_sphere",
            label="- embodiment-aware envelope",
            envelope="sphere",
            robot_radius=0.25,
        ),
        replace(
            full,
            key="static_dhocbf",
            label="- recoverable margin",
            constraint_mode="dhocbf",
            horizon_inflation_growth=0.0,
            use_tightening=False,
            rsm_base_margin=0.0,
            rsm_tau=0.0,
            rsm_brake_gain=0.0,
            rsm_terminal_margin=0.0,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
        ),
        replace(
            full,
            key="no_recoverability",
            label="- recoverability",
            constraint_mode="dhocbf",
            horizon_inflation_growth=0.0,
            use_tightening=False,
            rsm_base_margin=0.0,
            rsm_tau=0.0,
            rsm_brake_gain=0.0,
            rsm_terminal_margin=0.0,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
        ),
        replace(
            full,
            key="no_tightening",
            label="- horizon tightening",
            use_tightening=False,
            horizon_inflation_growth=0.0,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
        ),
        replace(
            full,
            key="instant_rsm",
            label="- sampled-data recursion",
            constraint_mode="rsm_instant",
            use_tightening=False,
            horizon_inflation_growth=0.0,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
            rsm_terminal_margin=0.0,
        ),
        replace(
            full,
            key="pointwise_gp",
            label="- Uniform confidence",
            confidence_inflation=0.012,
            horizon_inflation_growth=0.001,
            delta0=0.0,
            delta1=0.0,
            delta2=0.0,
        ),
    ]


def sensitivity_groups(sim: SimConfig) -> Dict[str, List[ControllerConfig]]:
    full = ControllerConfig(key="full", label="Full RSM-NMPC", dt=sim.dt, body_axes=sim.body_axes)
    return {
        "horizon": [
            replace(full, key="N4", label="N=4", horizon=4, ipopt_max_iter=70),
            replace(full, key="N8", label="N=8", horizon=8),
            replace(full, key="N12", label="N=12", horizon=12, ipopt_max_iter=110),
        ],
        "lambda": [
            replace(full, key="lambda5", label="lambda=5", lse_lambda=5.0),
            replace(full, key="lambda10", label="lambda=10", lse_lambda=10.0),
            replace(full, key="lambda20", label="lambda=20", lse_lambda=20.0),
        ],
        "gamma1": [
            replace(full, key="gamma01", label="gamma1=0.1", gamma1=0.10),
            replace(full, key="gamma03", label="gamma1=0.3", gamma1=0.30),
            replace(full, key="gamma05", label="gamma1=0.5", gamma1=0.50),
        ],
        "gp_length": [
            replace(full, key="ell02", label="ell=0.2m", confidence_inflation=0.025),
            replace(full, key="ell04", label="ell=0.4m", confidence_inflation=0.045),
            replace(full, key="ell08", label="ell=0.8m", confidence_inflation=0.070),
        ],
    }
