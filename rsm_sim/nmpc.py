"""CasADi NMPC controllers used by the simulation experiments."""

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import casadi as ca
import numpy as np

from .config import ControllerConfig, SimConfig
from .geometry import (
    aggregated_barrier_value,
    barrier_recursion_diagnostics,
    dynamic_recovery_reserve,
    full_body_clearance_components,
    recoverability_margin,
    recoverable_barrier_value,
    rsm_certificate,
    true_body_clearance,
)
from .scenarios import Obstacle, Scenario, reference_from_path


SUCCESS_STATUSES = {
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Search_Direction_Becomes_Too_Small",
    "Maximum_Iterations_Exceeded",
    "Maximum_CpuTime_Exceeded",
}


@dataclass
class SolveResult:
    control: np.ndarray
    feasible: bool
    solve_time_ms: float
    status: str
    warm_start: Optional[np.ndarray]
    min_constraint_margin: float
    predicted_states: Optional[np.ndarray] = None
    predicted_controls: Optional[np.ndarray] = None
    max_safety_slack: float = 0.0
    diagnostics: Optional[Dict[str, float]] = None


def step_dynamics(state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
    x, y, vx, vy, theta, omega = state
    ax, ay, alpha = control
    return np.array(
        [
            x + dt * vx + 0.5 * dt * dt * ax,
            y + dt * vy + 0.5 * dt * dt * ay,
            vx + dt * ax,
            vy + dt * ay,
            theta + dt * omega + 0.5 * dt * dt * alpha,
            omega + dt * alpha,
        ],
        dtype=float,
    )


def clip_control(control: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    out = np.array(control, dtype=float).copy()
    acc_norm = np.linalg.norm(out[:2])
    if acc_norm > cfg.a_max:
        out[:2] *= cfg.a_max / acc_norm
    out[2] = np.clip(out[2], -cfg.alpha_max, cfg.alpha_max)
    return out


class CasadiNMPC:
    nx = 6
    nu = 3

    def __init__(self, cfg: ControllerConfig, max_obstacles: int):
        self.cfg = cfg
        self.max_obstacles = max_obstacles
        self.has_safety_slack = cfg.constraint_mode in {"distance", "dhocbf", "rsm", "rsm_instant"}
        self.n_slack = cfg.horizon + 1 if self.has_safety_slack else 0
        self.n_state_dec = self.nx * (cfg.horizon + 1)
        self.n_control_dec = self.nu * cfg.horizon
        self.n_decision = self.n_state_dec + self.n_control_dec + self.n_slack
        self._build_solver()

    def _symbolic_step(self, state, control):
        dt = self.cfg.dt
        return ca.vertcat(
            state[0] + dt * state[2] + 0.5 * dt * dt * control[0],
            state[1] + dt * state[3] + 0.5 * dt * dt * control[1],
            state[2] + dt * control[0],
            state[3] + dt * control[1],
            state[4] + dt * state[5] + 0.5 * dt * dt * control[2],
            state[5] + dt * control[2],
        )

    def _build_solver(self) -> None:
        cfg = self.cfg
        n = cfg.horizon
        m = self.max_obstacles

        x0 = ca.SX.sym("x0", self.nx)
        u_prev = ca.SX.sym("u_prev", self.nu)
        ref = ca.SX.sym("ref", 2)
        centers = ca.SX.sym("centers", 2 * m)
        radii = ca.SX.sym("radii", m)
        infl = ca.SX.sym("infl", n + 1)
        params = ca.vertcat(x0, u_prev, ref, centers, radii, infl)

        x_vars = ca.SX.sym("X", self.nx, n + 1)
        u_vars = ca.SX.sym("U", self.nu, n)
        slack = ca.SX.sym("eps", self.n_slack) if self.has_safety_slack else ca.SX.zeros(0, 1)
        decision = ca.vertcat(
            *[x_vars[:, i] for i in range(n + 1)],
            *[u_vars[:, i] for i in range(n)],
            slack,
        )

        cost = 0
        constraints = []
        lower: List[float] = []
        upper: List[float] = []
        labels: List[str] = []

        def add_constraint(expr, lb: float, ub: float, label: str = "constraint") -> None:
            expr_vec = ca.reshape(expr, -1, 1)
            constraints.append(expr_vec)
            count = int(expr_vec.numel())
            lower.extend([lb] * count)
            upper.extend([ub] * count)
            labels.extend([label] * count)

        add_constraint(x_vars[:, 0] - x0, 0.0, 0.0, "initial_state")
        for i in range(n):
            add_constraint(x_vars[:, i + 1] - self._symbolic_step(x_vars[:, i], u_vars[:, i]), 0.0, 0.0, "dynamics")

        for i in range(n):
            u = u_vars[:, i]
            ns = x_vars[:, i + 1]
            du = u - (u_prev if i == 0 else u_vars[:, i - 1])
            progress_weight = 1.0 + 0.20 * float(i + 1) / max(float(n), 1.0)
            pos_err = ns[0:2] - ref
            cost += progress_weight * cfg.q_pos * ca.sumsqr(pos_err)
            cost += cfg.q_vel * ca.sumsqr(ns[2:4])
            cost += cfg.q_theta * ns[4] ** 2
            cost += cfg.r_acc * ca.sumsqr(u[0:2]) + cfg.r_alpha * u[2] ** 2
            cost += cfg.r_du * ca.sumsqr(du)
        cost += cfg.terminal_pos * ca.sumsqr(x_vars[:, n][0:2] - ref)

        if self.has_safety_slack:
            cost += cfg.safety_slack_weight * ca.sumsqr(slack)
            cost += cfg.safety_slack_linear * ca.sum1(slack)

        for i in range(1, n + 1):
            state_i = x_vars[:, i]
            add_constraint(state_i[2] ** 2 + state_i[3] ** 2, -ca.inf, cfg.v_max ** 2, "speed_limit")
            add_constraint(state_i[4], -cfg.theta_max, cfg.theta_max, "attitude_limit")
            add_constraint(state_i[5], -cfg.omega_max, cfg.omega_max, "rate_limit")

        def eps(i: int):
            return slack[i] if self.has_safety_slack else 0.0

        if cfg.constraint_mode == "distance":
            for i in range(1, n + 1):
                h = self._h_value(x_vars[:, i], i, centers, radii, infl)
                add_constraint(h + eps(i), 0.0, ca.inf, "distance")
        elif cfg.constraint_mode == "dhocbf":
            hs = [self._h_value(x_vars[:, i], i, centers, radii, infl) for i in range(n + 1)]
            self._add_sampled_barrier_constraints(hs, eps, add_constraint)
        elif cfg.constraint_mode in {"rsm", "rsm_instant"}:
            clearances = [self._h_value(x_vars[:, i], i, centers, radii, infl) for i in range(n + 1)]
            deltas = [
                self._recoverability_margin(x_vars[:, i], i, ref, centers, radii, infl)
                for i in range(n + 1)
            ]
            hs = [self._rsm_value(clearances[i], deltas[i]) for i in range(n + 1)]
            if cfg.constraint_mode == "rsm":
                self._add_sampled_barrier_constraints(hs, eps, add_constraint)
            else:
                for i, h in enumerate(hs):
                    add_constraint(h + eps(i), 0.0, ca.inf, "instant_h_R")
            if cfg.rsm_terminal_margin > 0.0:
                add_constraint(hs[-1] + eps(n), cfg.rsm_terminal_margin, ca.inf, "rsm_terminal")

        g = ca.vertcat(*constraints) if constraints else ca.SX.zeros(0, 1)
        nlp = {"x": decision, "p": params, "f": cost, "g": g}
        opts = {
            "verbose": False,
            "ipopt.print_level": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": cfg.ipopt_max_iter,
            "ipopt.max_cpu_time": cfg.ipopt_max_cpu_time,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 3,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_init": 1e-3,
            "print_time": 0,
        }
        tag = "%s_%x" % ("".join(ch if ch.isalnum() else "_" for ch in cfg.key), id(self) & 0xFFFFF)
        self.solver = ca.nlpsol("solver_" + tag, "ipopt", nlp, opts)
        self.g_fun = ca.Function("g_" + tag, [decision, params], [g])
        self.lbg = np.array([float(v) if v != -ca.inf else -np.inf for v in lower], dtype=float)
        self.ubg = np.array([float(v) if v != ca.inf else np.inf for v in upper], dtype=float)
        self.constraint_labels = np.array(labels, dtype=object)

        state_lb = [-np.inf] * self.n_state_dec
        state_ub = [np.inf] * self.n_state_dec
        control_lb = []
        control_ub = []
        for _ in range(n):
            control_lb += [-cfg.a_max, -cfg.a_max, -cfg.alpha_max]
            control_ub += [cfg.a_max, cfg.a_max, cfg.alpha_max]
        slack_lb = [0.0] * self.n_slack
        slack_ub = [cfg.safety_slack_max] * self.n_slack
        self.lbx = np.array(state_lb + control_lb + slack_lb, dtype=float)
        self.ubx = np.array(state_ub + control_ub + slack_ub, dtype=float)

    def _add_sampled_barrier_constraints(self, hs, eps, add_constraint) -> None:
        cfg = self.cfg
        n = cfg.horizon
        d0 = cfg.delta0 if cfg.use_tightening else 0.0
        d1 = cfg.delta1 if cfg.use_tightening else 0.0
        d2 = cfg.delta2 if cfg.use_tightening else 0.0

        for i, h in enumerate(hs):
            add_constraint(h + eps(i), i * d0, ca.inf, "psi0")
        psi1 = []
        for i in range(n):
            val = hs[i + 1] - hs[i] + cfg.gamma1 * hs[i]
            psi1.append(val)
            add_constraint(val + 0.5 * (eps(i) + eps(i + 1)), (i + 1) * d1, ca.inf, "psi1")
        for i in range(n - 1):
            val = psi1[i + 1] - psi1[i] + cfg.gamma2 * psi1[i]
            add_constraint(val + 0.5 * (eps(i + 1) + eps(i + 2)), (i + 1) * d2, ca.inf, "psi2")

    def _obs_sdf(self, point, stage: int, centers, radii, infl):
        cfg = self.cfg
        terms = []
        for j in range(self.max_obstacles):
            cx = centers[2 * j]
            cy = centers[2 * j + 1]
            r = radii[j]
            dist = ca.sqrt((point[0] - cx) ** 2 + (point[1] - cy) ** 2 + 1e-7) - r - infl[stage]
            terms.append(dist)
        z = [-cfg.obs_lse_lambda * t for t in terms]
        zmax = ca.mmax(ca.vertcat(*z))
        return -(ca.log(ca.sum1(ca.exp(ca.vertcat(*z) - zmax))) + zmax) / cfg.obs_lse_lambda

    def _support_offsets(self):
        cfg = self.cfg
        if cfg.envelope == "point":
            return [(0.0, 0.0)]
        if cfg.envelope == "sphere":
            angles = np.linspace(0.0, 2.0 * np.pi, int(cfg.body_constraint_samples), endpoint=False)
            return [(cfg.robot_radius * float(np.cos(a)), cfg.robot_radius * float(np.sin(a))) for a in angles]
        angles = np.linspace(0.0, 2.0 * np.pi, int(cfg.body_constraint_samples), endpoint=False)
        ax, ay = cfg.body_axes
        if cfg.envelope == "ellipse":
            return [(ax * float(np.cos(a)), ay * float(np.sin(a))) for a in angles]
        p = max(float(cfg.body_shape_exponent), 1.2)
        out = []
        for a in angles:
            c = float(np.cos(a))
            s = float(np.sin(a))
            out.append(
                (
                    ax * np.sign(c) * abs(c) ** (2.0 / p),
                    ay * np.sign(s) * abs(s) ** (2.0 / p),
                )
            )
        return out

    def _h_value(self, state, stage: int, centers, radii, infl):
        cfg = self.cfg
        if cfg.envelope == "point":
            return self._obs_sdf(state[0:2], stage, centers, radii, infl) - cfg.d_min
        if cfg.envelope == "sphere":
            return self._obs_sdf(state[0:2], stage, centers, radii, infl) - cfg.d_min - cfg.robot_radius

        ct = ca.cos(state[4])
        st = ca.sin(state[4])
        margins = []
        for lx, ly in self._support_offsets():
            bx = state[0] + ct * lx - st * ly
            by = state[1] + st * lx + ct * ly
            margins.append(self._obs_sdf(ca.vertcat(bx, by), stage, centers, radii, infl) - cfg.d_min - cfg.rho_b)
        z = [-cfg.lse_lambda * h for h in margins]
        zmax = ca.mmax(ca.vertcat(*z))
        return -(ca.log(ca.sum1(ca.exp(ca.vertcat(*z) - zmax))) + zmax) / cfg.lse_lambda

    def _reference_boundary_clearance(self, state, stage: int, ref, centers, radii, infl):
        cfg = self.cfg
        values = []
        ref_state = ca.vertcat(ref[0], ref[1], 0.0, 0.0, state[4], 0.0)
        values.append(self._h_value(ref_state, stage, centers, radii, infl))
        if cfg.rsm_gamma_boundary_radius > 1e-7 and cfg.rsm_gamma_boundary_samples > 0:
            for a in np.linspace(0.0, 2.0 * np.pi, int(cfg.rsm_gamma_boundary_samples), endpoint=False):
                dx = cfg.rsm_gamma_boundary_radius * float(np.cos(a))
                dy = cfg.rsm_gamma_boundary_radius * float(np.sin(a))
                b_state = ca.vertcat(ref[0] + dx, ref[1] + dy, 0.0, 0.0, state[4], 0.0)
                values.append(self._h_value(b_state, stage, centers, radii, infl))
        z = [-cfg.lse_lambda * v for v in values]
        zmax = ca.mmax(ca.vertcat(*z))
        return -(ca.log(ca.sum1(ca.exp(ca.vertcat(*z) - zmax))) + zmax) / cfg.lse_lambda

    def _recovery_reserve(self, state):
        cfg = self.cfg
        speed_sq = state[2] ** 2 + state[3] ** 2
        speed = ca.sqrt(speed_sq + 1e-7)
        return cfg.rsm_base_margin + cfg.rsm_tau * speed + cfg.rsm_brake_gain * speed_sq / (2.0 * max(cfg.a_max, 1e-6))

    def _recoverability_margin(self, state, stage: int, ref, centers, radii, infl):
        cfg = self.cfg
        ref_clearance = self._reference_boundary_clearance(state, stage, ref, centers, radii, infl)
        positive_ref_clearance = 0.5 * (ref_clearance + ca.sqrt(ref_clearance ** 2 + 1e-7))
        gamma = cfg.rsm_gamma0 + cfg.rsm_gamma_clearance_gain * positive_ref_clearance
        pos_err = state[0:2] - ref
        energy = (
            cfg.rsm_v_pos * ca.sumsqr(pos_err)
            + cfg.rsm_v_vel * (state[2] ** 2 + state[3] ** 2)
            + cfg.rsm_v_theta * state[4] ** 2
            + cfg.rsm_v_omega * state[5] ** 2
        )
        return gamma - energy - self._recovery_reserve(state)

    def _rsm_value(self, clearance, delta):
        cfg = self.cfg
        vals = ca.vertcat(clearance, delta)
        z = -cfg.rsm_lambda * vals
        zmax = ca.mmax(z)
        return -(ca.log(ca.sum1(ca.exp(z - zmax))) + zmax) / cfg.rsm_lambda

    def _pack_params(self, state: np.ndarray, u_prev: np.ndarray, ref: np.ndarray, obstacles: Iterable[Obstacle]) -> np.ndarray:
        centers = np.zeros((self.max_obstacles, 2), dtype=float)
        radii = np.zeros(self.max_obstacles, dtype=float)
        obs = list(obstacles)[: self.max_obstacles]
        for i, obstacle in enumerate(obs):
            centers[i] = obstacle.center
            radii[i] = obstacle.radius
        if len(obs) < self.max_obstacles:
            centers[len(obs) :] = np.array([1e3, 1e3])
        growth = self.cfg.horizon_inflation_growth if self.cfg.use_tightening else 0.0
        infl = np.array([self.cfg.confidence_inflation + i * growth for i in range(self.cfg.horizon + 1)], dtype=float)
        return np.concatenate((state, u_prev, ref, centers.reshape(-1), radii, infl))

    def _certificate_diagnostics(
        self,
        states: np.ndarray,
        ref: np.ndarray,
        obstacles: Iterable[Obstacle],
        slack: np.ndarray,
    ) -> Dict[str, float]:
        cfg = self.cfg
        obs = list(obstacles)[: self.max_obstacles]
        growth = cfg.horizon_inflation_growth if cfg.use_tightening else 0.0
        certs = []
        h_values = []
        for i, state in enumerate(states):
            infl = cfg.confidence_inflation + i * growth
            if cfg.constraint_mode in {"rsm", "rsm_instant"}:
                cert = rsm_certificate(
                    state,
                    ref,
                    obs,
                    cfg.body_axes,
                    cfg.body_constraint_samples,
                    cfg.d_min,
                    cfg.rho_b,
                    infl,
                    cfg.lse_lambda,
                    cfg.envelope,
                    cfg.robot_radius,
                    cfg.body_shape_exponent,
                    cfg.rsm_base_margin,
                    cfg.rsm_tau,
                    cfg.rsm_brake_gain,
                    cfg.a_max,
                    cfg.rsm_gamma0,
                    cfg.rsm_gamma_clearance_gain,
                    cfg.rsm_v_pos,
                    cfg.rsm_v_vel,
                    cfg.rsm_v_theta,
                    cfg.rsm_v_omega,
                    cfg.rsm_lambda,
                    cfg.rsm_gamma_boundary_radius,
                    cfg.rsm_gamma_boundary_samples,
                )
                h_values.append(cert.h_R)
            else:
                d_lower, c_body, support_count = full_body_clearance_components(
                    state,
                    obs,
                    cfg.body_axes,
                    cfg.body_constraint_samples,
                    cfg.d_min,
                    cfg.rho_b,
                    infl,
                    cfg.lse_lambda,
                    cfg.envelope,
                    cfg.robot_radius,
                    cfg.body_shape_exponent,
                )
                cert = {
                    "d_lower_min": d_lower,
                    "c_body": c_body,
                    "gamma": float("nan"),
                    "recovery_energy": float("nan"),
                    "dynamic_reserve": 0.0,
                    "delta": c_body,
                    "h_R": c_body,
                    "support_count": support_count,
                }
                h_values.append(c_body)
            certs.append(cert)

        def values(key: str) -> List[float]:
            out = []
            for cert in certs:
                if isinstance(cert, dict):
                    out.append(float(cert[key]))
                else:
                    out.append(float(getattr(cert, key)))
            return out

        def finite_min(vals: List[float]) -> float:
            arr = np.array(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            return float(np.min(finite)) if len(finite) else float("inf")

        def finite_max(vals: List[float]) -> float:
            arr = np.array(vals, dtype=float)
            finite = arr[np.isfinite(arr)]
            return float(np.max(finite)) if len(finite) else float("nan")

        recursion = barrier_recursion_diagnostics(
            h_values,
            cfg.gamma1,
            cfg.gamma2,
            cfg.delta0,
            cfg.delta1,
            cfg.delta2,
            cfg.use_tightening,
            slack,
        )
        diag: Dict[str, float] = {
            "min_pred_d_lower": finite_min(values("d_lower_min")),
            "min_pred_c_body": finite_min(values("c_body")),
            "min_pred_gamma": finite_min(values("gamma")),
            "max_pred_recovery_energy": finite_max(values("recovery_energy")),
            "max_pred_dynamic_reserve": finite_max(values("dynamic_reserve")),
            "min_pred_delta": finite_min(values("delta")),
            "min_pred_h_R": finite_min(values("h_R")),
            "support_count": float(max(values("support_count")) if certs else 0.0),
        }
        diag.update(recursion)
        return diag

    def _tracking_feedback_control(self, state: np.ndarray, u_prev: np.ndarray, ref: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        pos_err = np.array(ref, dtype=float) - np.array(state[:2], dtype=float)
        dist = float(np.linalg.norm(pos_err))
        if dist > 1e-6:
            desired_speed = min(cfg.v_max, 1.65 * dist)
            desired_vel = desired_speed * pos_err / dist
        else:
            desired_vel = np.zeros(2)
        acc = 2.15 * (desired_vel - state[2:4]) + 0.30 * pos_err
        alpha = -0.90 * state[5]
        cmd = clip_control(np.array([acc[0], acc[1], alpha], dtype=float), cfg)
        return clip_control(0.78 * cmd + 0.22 * np.array(u_prev, dtype=float), cfg)

    def _rollout_from_controls(self, state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.zeros((self.cfg.horizon + 1, self.nx), dtype=float)
        states[0] = np.array(state, dtype=float)
        for i in range(self.cfg.horizon):
            states[i + 1] = step_dynamics(states[i], controls[i], self.cfg.dt)
            states[i + 1, 2:4] = np.clip(states[i + 1, 2:4], -self.cfg.v_max, self.cfg.v_max)
            states[i + 1, 4] = np.clip(states[i + 1, 4], -self.cfg.theta_max, self.cfg.theta_max)
            states[i + 1, 5] = np.clip(states[i + 1, 5], -self.cfg.omega_max, self.cfg.omega_max)
        return states

    def _feedback_rollout(self, state: np.ndarray, u_prev: np.ndarray, ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        controls = np.zeros((self.cfg.horizon, self.nu), dtype=float)
        states = np.zeros((self.cfg.horizon + 1, self.nx), dtype=float)
        states[0] = np.array(state, dtype=float)
        last = np.array(u_prev, dtype=float)
        for i in range(self.cfg.horizon):
            controls[i] = self._tracking_feedback_control(states[i], last, ref)
            states[i + 1] = step_dynamics(states[i], controls[i], self.cfg.dt)
            states[i + 1, 2:4] = np.clip(states[i + 1, 2:4], -self.cfg.v_max, self.cfg.v_max)
            states[i + 1, 4] = np.clip(states[i + 1, 4], -self.cfg.theta_max, self.cfg.theta_max)
            states[i + 1, 5] = np.clip(states[i + 1, 5], -self.cfg.omega_max, self.cfg.omega_max)
            last = controls[i]
        return states, controls

    def _decision_from_parts(self, states: np.ndarray, controls: np.ndarray, slack: np.ndarray) -> np.ndarray:
        return np.concatenate((states.reshape(-1), controls.reshape(-1), slack.reshape(-1)))

    def _split_decision(self, decision: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = np.array(decision, dtype=float).reshape(-1)
        states = z[: self.n_state_dec].reshape((self.cfg.horizon + 1, self.nx))
        control_end = self.n_state_dec + self.n_control_dec
        controls = z[self.n_state_dec : control_end].reshape((self.cfg.horizon, self.nu))
        slack = z[control_end : control_end + self.n_slack] if self.n_slack else np.zeros(0, dtype=float)
        return states, controls, slack

    def _initial_guess(self, state: np.ndarray, u_prev: np.ndarray, ref: np.ndarray) -> np.ndarray:
        states, controls = self._feedback_rollout(state, u_prev, ref)
        slack = np.zeros(self.n_slack, dtype=float)
        return self._decision_from_parts(states, controls, slack)

    def _warm_started_guess(
        self,
        state: np.ndarray,
        u_prev: np.ndarray,
        ref: np.ndarray,
        warm_start: Optional[np.ndarray],
    ) -> np.ndarray:
        base_states, base_controls = self._feedback_rollout(state, u_prev, ref)
        base_slack = np.zeros(self.n_slack, dtype=float)
        if warm_start is None:
            return self._decision_from_parts(base_states, base_controls, base_slack)

        warm = np.array(warm_start, dtype=float).reshape(-1)
        if warm.size == self.n_decision:
            _, warm_controls, warm_slack = self._split_decision(warm)
            shifted_controls = np.vstack((warm_controls[1:], warm_controls[-1:]))
            shifted_slack = np.concatenate((warm_slack[1:], warm_slack[-1:])) if self.n_slack else base_slack
        elif warm.size == self.n_control_dec:
            warm_controls = warm.reshape((self.cfg.horizon, self.nu))
            shifted_controls = np.vstack((warm_controls[1:], warm_controls[-1:]))
            shifted_slack = base_slack
        else:
            return self._decision_from_parts(base_states, base_controls, base_slack)

        blend = float(np.clip(self.cfg.warm_start_blend, 0.0, 1.0))
        controls = blend * shifted_controls + (1.0 - blend) * base_controls
        for i in range(self.cfg.horizon):
            controls[i] = clip_control(controls[i], self.cfg)
        states = self._rollout_from_controls(state, controls)
        slack = np.clip(0.75 * shifted_slack + 0.25 * base_slack, 0.0, self.cfg.safety_slack_max)
        return self._decision_from_parts(states, controls, slack)

    def _finite_constraint_margin(self, gval: np.ndarray) -> float:
        if not len(gval):
            return 0.0
        lower = gval - self.lbg
        upper = self.ubg - gval
        finite_lower = lower[np.isfinite(lower)]
        finite_upper = upper[np.isfinite(upper)]
        margins = []
        if len(finite_lower):
            margins.append(float(np.min(finite_lower)))
        if len(finite_upper):
            margins.append(float(np.min(finite_upper)))
        return min(margins) if margins else 0.0

    def _constraint_margins_by_label(self, gval: np.ndarray) -> Dict[str, float]:
        if not len(gval):
            return {}
        lower = gval - self.lbg
        upper = self.ubg - gval
        out: Dict[str, float] = {}
        for label in sorted(set(self.constraint_labels.tolist())):
            mask = self.constraint_labels == label
            vals = []
            finite_lower = lower[mask][np.isfinite(lower[mask])]
            finite_upper = upper[mask][np.isfinite(upper[mask])]
            if len(finite_lower):
                vals.append(float(np.min(finite_lower)))
            if len(finite_upper):
                vals.append(float(np.min(finite_upper)))
            if vals:
                out[label] = min(vals)
        return out

    def solve(
        self,
        state: np.ndarray,
        u_prev: np.ndarray,
        ref: np.ndarray,
        obstacles: Iterable[Obstacle],
        warm_start: Optional[np.ndarray] = None,
    ) -> SolveResult:
        params = self._pack_params(state, u_prev, ref, obstacles)
        x_init = self._warm_started_guess(state, u_prev, ref, warm_start)
        x_init = np.minimum(np.maximum(x_init, self.lbx), self.ubx)

        start = time.perf_counter()
        status = "Exception"
        try:
            sol = self.solver(
                x0=x_init,
                lbx=self.lbx,
                ubx=self.ubx,
                lbg=self.lbg,
                ubg=self.ubg,
                p=params,
            )
            elapsed = 1000.0 * (time.perf_counter() - start)
            status = self.solver.stats().get("return_status", "Unknown")
            sol_x = np.array(sol["x"]).reshape(-1)
            gval = np.array(self.g_fun(sol_x, params)).reshape(-1)
            margin = self._finite_constraint_margin(gval)
            pred_states, pred_controls, slack = self._split_decision(sol_x)
            max_slack = float(np.max(slack)) if len(slack) else 0.0
            diagnostics = self._certificate_diagnostics(pred_states, ref, obstacles, slack)
            grouped_margins = self._constraint_margins_by_label(gval)
            for label, value in grouped_margins.items():
                diagnostics["ocp_%s_margin" % label] = value
            if "psi0" in grouped_margins:
                diagnostics["min_psi0_constraint"] = grouped_margins["psi0"]
            if "psi1" in grouped_margins:
                diagnostics["min_psi1_constraint"] = grouped_margins["psi1"]
            if "psi2" in grouped_margins:
                diagnostics["min_psi2_constraint"] = grouped_margins["psi2"]
            feasible = status in SUCCESS_STATUSES and margin >= -1e-4
            return SolveResult(
                control=clip_control(pred_controls[0], self.cfg),
                feasible=feasible,
                solve_time_ms=elapsed,
                status=status,
                warm_start=sol_x,
                min_constraint_margin=margin,
                predicted_states=pred_states,
                predicted_controls=pred_controls,
                max_safety_slack=max_slack,
                diagnostics=diagnostics,
            )
        except Exception as exc:
            elapsed = 1000.0 * (time.perf_counter() - start)
            guess = self._initial_guess(state, u_prev, ref)
            _, guess_controls, _ = self._split_decision(guess)
            return SolveResult(
                control=clip_control(guess_controls[0], self.cfg),
                feasible=False,
                solve_time_ms=elapsed,
                status=type(exc).__name__ if status == "Exception" else status,
                warm_start=None,
                min_constraint_margin=float("-inf"),
                predicted_states=None,
                predicted_controls=None,
                max_safety_slack=float("inf"),
                diagnostics={},
            )


def fallback_control(state: np.ndarray, ref: np.ndarray, cfg: ControllerConfig) -> np.ndarray:
    pos_err = np.array(ref, dtype=float) - np.array(state[:2], dtype=float)
    dist = float(np.linalg.norm(pos_err))
    if dist > 1e-6:
        desired_speed = min(cfg.v_max, 1.35 * dist)
        desired_vel = desired_speed * pos_err / dist
    else:
        desired_vel = np.zeros(2)
    acc = 1.85 * (desired_vel - state[2:4]) + 0.20 * pos_err
    alpha = -0.80 * state[5]
    return clip_control(np.array([acc[0], acc[1], alpha]), cfg)


def simulate_controller(
    scenario: Scenario,
    cfg: ControllerConfig,
    sim: SimConfig,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    controller = CasadiNMPC(cfg, sim.max_obstacles)
    eval_cfg = ControllerConfig(key="rsm_eval", label="RSM evaluation", dt=sim.dt, body_axes=sim.body_axes)
    state = scenario.start.copy()
    u_prev = np.zeros(3)
    warm = None

    states: List[np.ndarray] = [state.copy()]
    controls: List[np.ndarray] = []
    refs: List[np.ndarray] = []
    solve_times: List[float] = []
    feasible_flags: List[bool] = []
    slack_values: List[float] = []
    clearances: List[float] = []
    d_lowers: List[float] = []
    barriers: List[float] = []
    gammas: List[float] = []
    energies: List[float] = []
    deltas: List[float] = []
    recoverable_margins: List[float] = []

    collision = False
    reached = False

    for _ in range(sim.max_steps):
        ref = reference_from_path(state[:2], scenario.path, scenario.goal, sim.lookahead)
        result = controller.solve(state, u_prev, ref, scenario.obstacles, warm)
        solve_times.append(result.solve_time_ms)
        feasible_flags.append(result.feasible)
        slack_values.append(result.max_safety_slack)
        warm = result.warm_start if result.warm_start is not None else None
        control = result.control if result.feasible else fallback_control(state, ref, cfg)

        noise = np.array(
            [
                rng.normal(scale=sim.process_noise_acc),
                rng.normal(scale=sim.process_noise_acc),
                rng.normal(scale=sim.process_noise_alpha),
            ]
        )
        exec_control = clip_control(control + noise, cfg)
        state = step_dynamics(state, exec_control, sim.dt)
        state[2:4] = np.clip(state[2:4], -cfg.v_max * 1.1, cfg.v_max * 1.1)
        state[4] = np.clip(state[4], -cfg.theta_max * 1.2, cfg.theta_max * 1.2)
        state[5] = np.clip(state[5], -cfg.omega_max * 1.2, cfg.omega_max * 1.2)

        clearance = true_body_clearance(
            state,
            scenario.obstacles,
            sim.body_axes,
            sim.body_samples,
            eval_cfg.envelope,
            eval_cfg.robot_radius,
            eval_cfg.body_shape_exponent,
        )
        cert = rsm_certificate(
            state,
            ref,
            scenario.obstacles,
            sim.body_axes,
            eval_cfg.body_constraint_samples,
            eval_cfg.d_min,
            eval_cfg.rho_b,
            eval_cfg.confidence_inflation,
            eval_cfg.lse_lambda,
            eval_cfg.envelope,
            eval_cfg.robot_radius,
            eval_cfg.body_shape_exponent,
            eval_cfg.rsm_base_margin,
            eval_cfg.rsm_tau,
            eval_cfg.rsm_brake_gain,
            eval_cfg.a_max,
            eval_cfg.rsm_gamma0,
            eval_cfg.rsm_gamma_clearance_gain,
            eval_cfg.rsm_v_pos,
            eval_cfg.rsm_v_vel,
            eval_cfg.rsm_v_theta,
            eval_cfg.rsm_v_omega,
            eval_cfg.rsm_lambda,
            eval_cfg.rsm_gamma_boundary_radius,
            eval_cfg.rsm_gamma_boundary_samples,
        )
        barrier = cert.c_body
        states.append(state.copy())
        controls.append(exec_control.copy())
        refs.append(ref.copy())
        clearances.append(clearance)
        d_lowers.append(cert.d_lower_min)
        barriers.append(barrier)
        gammas.append(cert.gamma)
        energies.append(cert.recovery_energy)
        deltas.append(cert.delta)
        recoverable_margins.append(cert.h_R)
        u_prev = exec_control

        if clearance < -1e-3:
            collision = True
            break
        if np.linalg.norm(state[:2] - scenario.goal) <= sim.reach_radius:
            reached = True
            break

    states_arr = np.vstack(states)
    controls_arr = np.vstack(controls) if controls else np.zeros((0, 3))
    speeds = np.linalg.norm(states_arr[:, 2:4], axis=1)
    active = speeds[(speeds > 0.10) & (np.linalg.norm(states_arr[:, :2] - scenario.goal, axis=1) > sim.reach_radius)]
    trial_steps = max(1, len(states_arr) - 1)
    metrics = {
        "controller": cfg.key,
        "controller_label": cfg.label,
        "scenario": scenario.name,
        "seed": seed,
        "success": float(reached and not collision),
        "collision": float(collision),
        "reached": float(reached),
        "steps": float(trial_steps),
        "duration_s": float(trial_steps * sim.dt),
        "min_clearance_m": float(np.min(clearances) if clearances else np.nan),
        "min_d_lower": float(np.min(d_lowers) if d_lowers else np.nan),
        "min_c_body": float(np.min(barriers) if barriers else np.nan),
        "min_barrier": float(np.min(barriers) if barriers else np.nan),
        "min_delta": float(np.min(deltas) if deltas else np.nan),
        "min_h_R": float(np.min(recoverable_margins) if recoverable_margins else np.nan),
        "min_recoverable_margin": float(np.min(recoverable_margins) if recoverable_margins else np.nan),
        "avg_speed_mps": float(np.mean(speeds)),
        "min_active_speed_mps": float(np.min(active) if len(active) else 0.0),
        "feasibility_rate": float(np.mean(feasible_flags) if feasible_flags else 0.0),
        "max_safety_slack": float(np.max(slack_values) if slack_values else 0.0),
        "mean_safety_slack": float(np.mean(slack_values) if slack_values else 0.0),
        "compute_ms": float(np.mean(solve_times) if solve_times else np.nan),
        "compute_ms_max": float(np.max(solve_times) if solve_times else np.nan),
    }
    traces = {
        "states": states_arr,
        "controls": controls_arr,
        "refs": np.vstack(refs) if refs else np.zeros((0, 2)),
        "clearance": np.array(clearances),
        "d_lower": np.array(d_lowers),
        "barrier": np.array(barriers),
        "gamma": np.array(gammas),
        "recovery_energy": np.array(energies),
        "delta": np.array(deltas),
        "recoverable_margin": np.array(recoverable_margins),
        "feasible": np.array(feasible_flags, dtype=float),
        "safety_slack": np.array(slack_values, dtype=float),
        "solve_ms": np.array(solve_times),
    }
    return metrics, traces
