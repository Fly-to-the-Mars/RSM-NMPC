"""Geometry helpers for support-point and clearance evaluation."""

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class CertificateDiagnostics:
    """Numerical RSM certificate values for one state/reference pair."""

    d_lower_min: float
    c_body: float
    gamma: float
    recovery_energy: float
    dynamic_reserve: float
    delta: float
    h_R: float
    support_count: int


def rot2(theta: float) -> np.ndarray:
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def ellipse_support_points(pos: np.ndarray, theta: float, axes: Tuple[float, float], count: int) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    local = np.column_stack((axes[0] * np.cos(ang), axes[1] * np.sin(ang)))
    return pos.reshape(1, 2) + local @ rot2(theta).T


def superellipse_support_points(
    pos: np.ndarray,
    theta: float,
    axes: Tuple[float, float],
    count: int,
    exponent: float = 4.0,
) -> np.ndarray:
    """Boundary support samples for a 2-D superellipsoidal body section.

    This is the planar cross-section of the quadrotor envelope used by the
    navigation-layer certificate. `exponent=2` recovers an ellipse; larger
    exponents are closer to a rounded rectangle.
    """
    p = max(float(exponent), 1.2)
    ang = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    c = np.cos(ang)
    s = np.sin(ang)
    local = np.column_stack(
        (
            axes[0] * np.sign(c) * np.abs(c) ** (2.0 / p),
            axes[1] * np.sign(s) * np.abs(s) ** (2.0 / p),
        )
    )
    return pos.reshape(1, 2) + local @ rot2(theta).T


def circle_support_points(pos: np.ndarray, radius: float, count: int) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    return pos.reshape(1, 2) + radius * np.column_stack((np.cos(ang), np.sin(ang)))


def body_support_points(
    state: np.ndarray,
    axes: Tuple[float, float],
    samples: int,
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> np.ndarray:
    """Return S(q): sampled body boundary points b_j = p + R(q) xi_j."""
    if envelope == "point":
        return np.asarray(state[:2], dtype=float).reshape(1, 2)
    if envelope == "sphere":
        return circle_support_points(state[:2], robot_radius, samples)
    if envelope == "ellipse":
        return ellipse_support_points(state[:2], state[4], axes, samples)
    return superellipse_support_points(state[:2], state[4], axes, samples, shape_exponent)


def obstacle_sdf(points: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(points)
    if len(centers) == 0:
        return np.full(len(pts), np.inf)
    d = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2) - radii[None, :]
    return d.min(axis=1)


def obstacle_arrays(obstacles: Iterable, confidence_inflation: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    obs = list(obstacles)
    if not obs:
        return np.zeros((0, 2), dtype=float), np.zeros(0, dtype=float)
    centers = np.array([o.center for o in obs], dtype=float)
    radii = np.array([o.radius + confidence_inflation for o in obs], dtype=float)
    return centers, radii


def true_body_clearance(
    state: np.ndarray,
    obstacles: Iterable,
    axes: Tuple[float, float],
    samples: int,
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> float:
    obs = list(obstacles)
    if not obs:
        return float("inf")
    centers = np.array([o.center for o in obs], dtype=float)
    radii = np.array([o.radius for o in obs], dtype=float)
    support = body_support_points(state, axes, samples, envelope, robot_radius, shape_exponent)
    return float(obstacle_sdf(support, centers, radii).min())


def softmin(values: List[float], sharpness: float) -> float:
    arr = np.array(values, dtype=float)
    z = -sharpness * arr
    zmax = z.max()
    return float(-(math.log(np.exp(z - zmax).sum()) + zmax) / sharpness)


def aggregated_barrier_value(
    state: np.ndarray,
    obstacles: Iterable,
    axes: Tuple[float, float],
    samples: int,
    d_min: float,
    rho_b: float,
    confidence_inflation: float,
    lse_lambda: float,
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> float:
    obs = list(obstacles)
    if not obs:
        return float("inf")
    centers, radii = obstacle_arrays(obs, confidence_inflation)
    support = body_support_points(state, axes, samples, envelope, robot_radius, shape_exponent)
    margins = obstacle_sdf(support, centers, radii) - d_min - rho_b
    return softmin(margins.tolist(), lse_lambda)


def full_body_clearance_components(
    state: np.ndarray,
    obstacles: Iterable,
    axes: Tuple[float, float],
    samples: int,
    d_min: float,
    rho_b: float,
    confidence_inflation: float,
    lse_lambda: float,
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> Tuple[float, float, int]:
    """Compute d_lower and c_k(q) from the perception-aware body envelope."""
    obs = list(obstacles)
    support = body_support_points(state, axes, samples, envelope, robot_radius, shape_exponent)
    if not obs:
        return float("inf"), float("inf"), len(support)
    centers, inflated_radii = obstacle_arrays(obs, confidence_inflation)
    d_lower = obstacle_sdf(support, centers, inflated_radii)
    margins = d_lower - d_min - rho_b
    return float(np.min(d_lower)), softmin(margins.tolist(), lse_lambda), len(support)


def dynamic_recovery_reserve(
    state: np.ndarray,
    base_margin: float,
    tau: float,
    brake_gain: float,
    a_max: float,
) -> float:
    """Conservative state-dependent clearance reserve for recoverability."""
    speed = float(np.linalg.norm(state[2:4]))
    return float(base_margin + tau * speed + brake_gain * speed * speed / (2.0 * max(a_max, 1e-6)))


def recovery_energy(
    state: np.ndarray,
    ref: np.ndarray,
    w_pos: float,
    w_vel: float,
    w_theta: float,
    w_omega: float,
) -> float:
    pos_err = np.asarray(state[:2], dtype=float) - np.asarray(ref[:2], dtype=float)
    return float(
        w_pos * float(np.dot(pos_err, pos_err))
        + w_vel * float(np.dot(state[2:4], state[2:4]))
        + w_theta * float(state[4] * state[4])
        + w_omega * float(state[5] * state[5])
    )


def recovery_threshold(reference_clearance: float, gamma0: float, clearance_gain: float) -> float:
    if not np.isfinite(reference_clearance):
        return float("inf")
    positive_clearance = 0.5 * (reference_clearance + math.sqrt(reference_clearance * reference_clearance + 1e-9))
    return float(gamma0 + clearance_gain * positive_clearance)


def reference_boundary_clearance(
    state: np.ndarray,
    ref: np.ndarray,
    obstacles: Iterable,
    axes: Tuple[float, float],
    samples: int,
    d_min: float,
    rho_b: float,
    confidence_inflation: float,
    lse_lambda: float,
    boundary_radius: float,
    boundary_samples: int,
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> float:
    """Conservative clearance used to build Gamma_k(r).

    The reference set is probed at the center and on a small boundary around
    the reference. Taking the minimum makes Gamma_k(r) depend on the clearance
    of a local recovery tube, rather than on a single point.
    """
    offsets = [np.zeros(2, dtype=float)]
    if boundary_radius > 1e-6 and boundary_samples > 0:
        for a in np.linspace(0.0, 2.0 * math.pi, int(boundary_samples), endpoint=False):
            offsets.append(boundary_radius * np.array([math.cos(a), math.sin(a)], dtype=float))
    values = []
    for offset in offsets:
        ref_state = np.array(state, dtype=float).copy()
        ref_state[:2] = np.asarray(ref[:2], dtype=float) + offset
        ref_state[2:4] = 0.0
        ref_state[5] = 0.0
        _, c_body, _ = full_body_clearance_components(
            ref_state,
            obstacles,
            axes,
            samples,
            d_min,
            rho_b,
            confidence_inflation,
            lse_lambda,
            envelope,
            robot_radius,
            shape_exponent,
        )
        values.append(c_body)
    finite = [v for v in values if np.isfinite(v)]
    return softmin(finite, lse_lambda) if finite else float("inf")


def recoverability_margin(
    state: np.ndarray,
    ref: np.ndarray,
    reference_clearance: float,
    base_margin: float,
    tau: float,
    brake_gain: float,
    a_max: float,
    gamma0: float,
    clearance_gain: float,
    w_pos: float,
    w_vel: float,
    w_theta: float,
    w_omega: float,
) -> float:
    reserve = dynamic_recovery_reserve(state, base_margin, tau, brake_gain, a_max)
    energy = recovery_energy(state, ref, w_pos, w_vel, w_theta, w_omega)
    return recovery_threshold(reference_clearance, gamma0, clearance_gain) - energy - reserve


def recoverable_barrier_value(clearance: float, delta: float, sharpness: float) -> float:
    vals = [v for v in (clearance, delta) if np.isfinite(v)]
    if not vals:
        return float("inf")
    if len(vals) == 1:
        return float(vals[0])
    return softmin(vals, sharpness)


def rsm_certificate(
    state: np.ndarray,
    ref: np.ndarray,
    obstacles: Iterable,
    axes: Tuple[float, float],
    samples: int,
    d_min: float,
    rho_b: float,
    confidence_inflation: float,
    lse_lambda: float,
    envelope: str,
    robot_radius: float,
    shape_exponent: float,
    base_margin: float,
    tau: float,
    brake_gain: float,
    a_max: float,
    gamma0: float,
    clearance_gain: float,
    w_pos: float,
    w_vel: float,
    w_theta: float,
    w_omega: float,
    rsm_lambda: float,
    gamma_boundary_radius: float = 0.0,
    gamma_boundary_samples: int = 0,
) -> CertificateDiagnostics:
    d_lower, c_body, support_count = full_body_clearance_components(
        state,
        obstacles,
        axes,
        samples,
        d_min,
        rho_b,
        confidence_inflation,
        lse_lambda,
        envelope,
        robot_radius,
        shape_exponent,
    )
    ref_clearance = reference_boundary_clearance(
        state,
        ref,
        obstacles,
        axes,
        samples,
        d_min,
        rho_b,
        confidence_inflation,
        lse_lambda,
        gamma_boundary_radius,
        gamma_boundary_samples,
        envelope,
        robot_radius,
        shape_exponent,
    )
    gamma = recovery_threshold(ref_clearance, gamma0, clearance_gain)
    energy = recovery_energy(state, ref, w_pos, w_vel, w_theta, w_omega)
    reserve = dynamic_recovery_reserve(state, base_margin, tau, brake_gain, a_max)
    delta = gamma - energy - reserve
    h_r = recoverable_barrier_value(c_body, delta, rsm_lambda)
    return CertificateDiagnostics(
        d_lower_min=d_lower,
        c_body=c_body,
        gamma=gamma,
        recovery_energy=energy,
        dynamic_reserve=reserve,
        delta=delta,
        h_R=h_r,
        support_count=support_count,
    )


def barrier_recursion_diagnostics(
    h_values: Iterable[float],
    gamma1: float,
    gamma2: float,
    delta0: float,
    delta1: float,
    delta2: float,
    use_tightening: bool = True,
    slacks: Iterable[float] = (),
) -> Dict[str, float]:
    hs = np.array(list(h_values), dtype=float)
    if hs.size == 0:
        return {
            "min_psi0": float("inf"),
            "min_psi1": float("inf"),
            "min_psi2": float("inf"),
            "min_psi0_constraint": float("inf"),
            "min_psi1_constraint": float("inf"),
            "min_psi2_constraint": float("inf"),
        }
    slack = np.zeros_like(hs)
    raw_slacks = list(slacks)
    if raw_slacks:
        raw = np.array(raw_slacks, dtype=float)
        slack[: min(len(raw), len(slack))] = raw[: min(len(raw), len(slack))]
    d0 = delta0 if use_tightening else 0.0
    d1 = delta1 if use_tightening else 0.0
    d2 = delta2 if use_tightening else 0.0
    psi0 = hs
    psi0_constraints = np.array([hs[i] + slack[i] - i * d0 for i in range(len(hs))], dtype=float)
    if len(hs) >= 2:
        psi1 = np.array([hs[i + 1] - hs[i] + gamma1 * hs[i] for i in range(len(hs) - 1)], dtype=float)
        psi1_constraints = np.array(
            [psi1[i] + 0.5 * (slack[i] + slack[i + 1]) - (i + 1) * d1 for i in range(len(psi1))],
            dtype=float,
        )
    else:
        psi1 = np.array([float("inf")])
        psi1_constraints = psi1.copy()
    if len(psi1) >= 2:
        psi2 = np.array([psi1[i + 1] - psi1[i] + gamma2 * psi1[i] for i in range(len(psi1) - 1)], dtype=float)
        psi2_constraints = np.array(
            [psi2[i] + 0.5 * (slack[i + 1] + slack[i + 2]) - (i + 1) * d2 for i in range(len(psi2))],
            dtype=float,
        )
    else:
        psi2 = np.array([float("inf")])
        psi2_constraints = psi2.copy()

    def finite_min(arr: np.ndarray) -> float:
        finite = arr[np.isfinite(arr)]
        return float(np.min(finite)) if len(finite) else float("inf")

    return {
        "min_psi0": finite_min(psi0),
        "min_psi1": finite_min(psi1),
        "min_psi2": finite_min(psi2),
        "min_psi0_constraint": finite_min(psi0_constraints),
        "min_psi1_constraint": finite_min(psi1_constraints),
        "min_psi2_constraint": finite_min(psi2_constraints),
    }


def polyline_length(path: np.ndarray) -> np.ndarray:
    if len(path) == 0:
        return np.array([0.0])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))
