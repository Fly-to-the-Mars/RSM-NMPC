#!/usr/bin/env python3
"""TV-DHOCBF controller running on SUPER's MARSIM/perfect_drone interface."""

import heapq
import csv
import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rospy
from geometry_msgs.msg import Point, PoseStamped, Vector3
from nav_msgs.msg import Odometry, Path as PathMsg
from quadrotor_msgs.msg import PositionCommand
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from sim_validation.config import ControllerConfig, SimConfig, ablation_controllers, baseline_controllers
from sim_validation.geometry import (
    aggregated_barrier_value,
    barrier_recursion_diagnostics,
    body_support_points,
    dynamic_recovery_reserve,
    ellipse_support_points,
    full_body_clearance_components,
    reference_boundary_clearance,
    recoverability_margin,
    recoverable_barrier_value,
    recovery_energy,
    recovery_threshold,
    rsm_certificate,
)
from sim_validation.nmpc import CasadiNMPC, clip_control, step_dynamics
from sim_validation.scenarios import Obstacle


FRAME_ID = "world"


def header() -> Header:
    return Header(stamp=rospy.Time.now(), frame_id=FRAME_ID)


def pt(x: float, y: float, z: float = 0.0) -> Point:
    return Point(float(x), float(y), float(z))


def marker_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    return ColorRGBA(float(r), float(g), float(b), float(a))


def dynamic_lifetime() -> rospy.Duration:
    return rospy.Duration(0.35)


def quat_from_yaw(yaw: float):
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    return 0.0, 0.0, qz, qw


def yaw_from_quat(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def path_msg(points: np.ndarray, z: float) -> PathMsg:
    msg = PathMsg()
    msg.header = header()
    for row in points:
        pose = PoseStamped()
        pose.header = msg.header
        pose.pose.position = pt(row[0], row[1], z)
        pose.pose.orientation.w = 1.0
        msg.poses.append(pose)
    return msg


def line_marker(ns: str, mid: int, points: np.ndarray, color: ColorRGBA, width: float, z: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.LINE_STRIP
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = width
    msg.color = color
    msg.points = [pt(p[0], p[1], z) for p in points]
    msg.lifetime = dynamic_lifetime()
    return msg


def line_marker_3d(ns: str, mid: int, points: np.ndarray, color: ColorRGBA, width: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.LINE_STRIP
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = width
    msg.color = color
    msg.points = [pt(p[0], p[1], p[2]) for p in points]
    msg.lifetime = dynamic_lifetime()
    return msg


def line_list_marker(ns: str, mid: int, points: np.ndarray, color: ColorRGBA, width: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.LINE_LIST
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = width
    msg.color = color
    msg.points = [pt(p[0], p[1], p[2]) for p in points]
    msg.lifetime = dynamic_lifetime()
    return msg


def sphere_list(ns: str, mid: int, points: np.ndarray, color: ColorRGBA, scale: float, z: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.SPHERE_LIST
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = scale
    msg.scale.y = scale
    msg.scale.z = scale
    msg.color = color
    msg.points = [pt(p[0], p[1], z) for p in points]
    msg.lifetime = dynamic_lifetime()
    return msg


def sphere_list_xyz(ns: str, mid: int, points: np.ndarray, color: ColorRGBA, scale: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.SPHERE_LIST
    msg.action = Marker.ADD
    msg.pose.orientation.w = 1.0
    msg.scale.x = scale
    msg.scale.y = scale
    msg.scale.z = scale
    msg.color = color
    msg.points = [pt(p[0], p[1], p[2]) for p in points]
    msg.lifetime = dynamic_lifetime()
    return msg


def ellipsoid_marker(ns: str, mid: int, xy: np.ndarray, yaw: float, axes: Tuple[float, float], z: float, color: ColorRGBA, height: float) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.SPHERE
    msg.action = Marker.ADD
    msg.pose.position = pt(xy[0], xy[1], z)
    qx, qy, qz, qw = quat_from_yaw(yaw)
    msg.pose.orientation.x = qx
    msg.pose.orientation.y = qy
    msg.pose.orientation.z = qz
    msg.pose.orientation.w = qw
    msg.scale.x = 2.0 * float(axes[0])
    msg.scale.y = 2.0 * float(axes[1])
    msg.scale.z = height
    msg.color = color
    msg.lifetime = dynamic_lifetime()
    return msg


def text_marker(ns: str, mid: int, text: str, xyz: Tuple[float, float, float]) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.TEXT_VIEW_FACING
    msg.action = Marker.ADD
    msg.pose.position = pt(*xyz)
    msg.pose.orientation.w = 1.0
    msg.scale.z = 0.65
    msg.color = marker_color(0.05, 0.05, 0.05, 1.0)
    msg.text = text
    msg.lifetime = dynamic_lifetime()
    return msg


def cylinder_marker(ns: str, mid: int, xy: np.ndarray, radius: float, z: float, color: ColorRGBA, height: float = 0.08) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.CYLINDER
    msg.action = Marker.ADD
    msg.pose.position = pt(xy[0], xy[1], z)
    msg.pose.orientation.w = 1.0
    msg.scale.x = 2.0 * float(radius)
    msg.scale.y = 2.0 * float(radius)
    msg.scale.z = float(height)
    msg.color = color
    msg.lifetime = dynamic_lifetime()
    return msg


def cube_marker(ns: str, mid: int, xyz: Tuple[float, float, float], scale: Tuple[float, float, float], color: ColorRGBA) -> Marker:
    msg = Marker()
    msg.header = header()
    msg.ns = ns
    msg.id = mid
    msg.type = Marker.CUBE
    msg.action = Marker.ADD
    msg.pose.position = pt(*xyz)
    msg.pose.orientation.w = 1.0
    msg.scale.x = float(scale[0])
    msg.scale.y = float(scale[1])
    msg.scale.z = float(scale[2])
    msg.color = color
    msg.lifetime = dynamic_lifetime()
    return msg


def get_controller(key: str, sim: SimConfig) -> ControllerConfig:
    mapping: Dict[str, ControllerConfig] = {c.key: c for c in baseline_controllers(sim)}
    mapping.update({c.key: c for c in ablation_controllers(sim)})
    cfg = mapping.get(key)
    if cfg is None:
        raise RuntimeError("Unknown controller key: %s" % key)
    gamma1 = float(rospy.get_param("~gamma1", cfg.gamma1))
    gamma2 = float(rospy.get_param("~gamma2", cfg.gamma2))
    if cfg.constraint_mode in {"rsm", "rsm_instant"}:
        gamma1 = float(rospy.get_param("~rsm_gamma1", gamma1))
        gamma2 = float(rospy.get_param("~rsm_gamma2", gamma2))
    # SUPER maps are denser than the lightweight validation scenes, so keep the
    # horizon modest while allowing more aggressive translational motion.
    return replace(
        cfg,
        horizon=int(rospy.get_param("~horizon", cfg.horizon)),
        dt=float(rospy.get_param("~nmpc_dt", 0.12)),
        d_min=float(rospy.get_param("~d_min", cfg.d_min)),
        rho_b=float(rospy.get_param("~rho_b", cfg.rho_b)),
        body_shape_exponent=float(rospy.get_param("~body_shape_exponent", cfg.body_shape_exponent)),
        body_constraint_samples=int(rospy.get_param("~body_constraint_samples", cfg.body_constraint_samples)),
        gamma1=gamma1,
        gamma2=gamma2,
        lse_lambda=float(rospy.get_param("~lse_lambda", cfg.lse_lambda)),
        obs_lse_lambda=float(rospy.get_param("~obs_lse_lambda", cfg.obs_lse_lambda)),
        v_max=float(rospy.get_param("~v_max", 4.0)),
        a_max=float(rospy.get_param("~a_max", 7.0)),
        theta_max=float(rospy.get_param("~theta_max", cfg.theta_max)),
        omega_max=float(rospy.get_param("~omega_max", cfg.omega_max)),
        confidence_inflation=float(rospy.get_param("~confidence_inflation", cfg.confidence_inflation)),
        horizon_inflation_growth=float(rospy.get_param("~horizon_inflation_growth", cfg.horizon_inflation_growth)),
        rsm_base_margin=float(rospy.get_param("~rsm_base_margin", cfg.rsm_base_margin)),
        rsm_tau=float(rospy.get_param("~rsm_tau", cfg.rsm_tau)),
        rsm_brake_gain=float(rospy.get_param("~rsm_brake_gain", cfg.rsm_brake_gain)),
        rsm_terminal_margin=float(rospy.get_param("~rsm_terminal_margin", cfg.rsm_terminal_margin)),
        rsm_lambda=float(rospy.get_param("~rsm_lambda", cfg.rsm_lambda)),
        rsm_gamma0=float(rospy.get_param("~rsm_gamma0", cfg.rsm_gamma0)),
        rsm_gamma_clearance_gain=float(rospy.get_param("~rsm_gamma_clearance_gain", cfg.rsm_gamma_clearance_gain)),
        rsm_gamma_boundary_radius=float(rospy.get_param("~rsm_gamma_boundary_radius", cfg.rsm_gamma_boundary_radius)),
        rsm_gamma_boundary_samples=int(rospy.get_param("~rsm_gamma_boundary_samples", cfg.rsm_gamma_boundary_samples)),
        rsm_v_pos=float(rospy.get_param("~rsm_v_pos", cfg.rsm_v_pos)),
        rsm_v_vel=float(rospy.get_param("~rsm_v_vel", cfg.rsm_v_vel)),
        rsm_v_theta=float(rospy.get_param("~rsm_v_theta", cfg.rsm_v_theta)),
        rsm_v_omega=float(rospy.get_param("~rsm_v_omega", cfg.rsm_v_omega)),
        safety_slack_weight=float(rospy.get_param("~safety_slack_weight", cfg.safety_slack_weight)),
        safety_slack_linear=float(rospy.get_param("~safety_slack_linear", cfg.safety_slack_linear)),
        safety_slack_max=float(rospy.get_param("~safety_slack_max", cfg.safety_slack_max)),
        ipopt_max_iter=int(rospy.get_param("~ipopt_max_iter", cfg.ipopt_max_iter)),
        ipopt_max_cpu_time=float(rospy.get_param("~ipopt_max_cpu_time", cfg.ipopt_max_cpu_time)),
    )


def is_rsm_mode(mode: str) -> bool:
    return mode in {"rsm", "rsm_instant"}


def cloud_to_array(msg: PointCloud2, max_points: int) -> np.ndarray:
    points = []
    for i, p in enumerate(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)):
        if i % 2 == 0:
            points.append((p[0], p[1], p[2]))
        if len(points) >= max_points:
            break
    if not points:
        return np.zeros((0, 3), dtype=float)
    return np.array(points, dtype=float)


def extract_obstacles_from_cloud(
    cloud: np.ndarray,
    pos_xy: np.ndarray,
    flight_z: float,
    max_obstacles: int,
    range_xy: float,
    grid: float,
    obstacle_radius: float,
    near_ignore: float,
) -> List[Obstacle]:
    if len(cloud) == 0:
        return []
    rel = cloud[:, :2] - pos_xy.reshape(1, 2)
    dist = np.linalg.norm(rel, axis=1)
    zmask = np.abs(cloud[:, 2] - flight_z) <= 1.4
    mask = (dist <= range_xy) & (dist >= near_ignore) & zmask
    pts = cloud[mask]
    if len(pts) == 0:
        return []
    buckets: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for p in pts:
        key = (int(round(p[0] / grid)), int(round(p[1] / grid)))
        buckets.setdefault(key, []).append(p[:2])
    centers = []
    for values in buckets.values():
        arr = np.vstack(values)
        c = arr.mean(axis=0)
        centers.append((float(np.linalg.norm(c - pos_xy)), c))
    centers.sort(key=lambda item: item[0])
    return [Obstacle(c, obstacle_radius) for _, c in centers[:max_obstacles]]


def local_point_clearance(
    cloud: np.ndarray,
    state: np.ndarray,
    flight_z: float,
    near_ignore: float,
    range_xy: float,
    body_axes: Tuple[float, float],
    envelope: str = "superellipse",
    robot_radius: float = 0.25,
    shape_exponent: float = 4.0,
) -> float:
    if len(cloud) == 0:
        return float("inf")
    rel = cloud[:, :2] - state[:2].reshape(1, 2)
    dist = np.linalg.norm(rel, axis=1)
    zmask = np.abs(cloud[:, 2] - flight_z) <= 1.4
    mask = (dist >= near_ignore) & (dist <= range_xy) & zmask
    pts = cloud[mask, :2]
    if len(pts) == 0:
        return float("inf")
    support = body_support_points(state, body_axes, 36, envelope, robot_radius, shape_exponent)
    return float(np.min(np.linalg.norm(pts[:, None, :] - support[None, :, :], axis=2)))


def plan_cloud_path(
    start: np.ndarray,
    goal: np.ndarray,
    cloud: np.ndarray,
    z: float,
    res: float,
    margin: float,
    max_extent: float,
    boundary_padding: float = 1.5,
    center_weight: float = 0.0,
    center_deadband: float = 0.0,
) -> np.ndarray:
    if len(cloud) == 0:
        return np.vstack((start, goal))
    focus_min = np.minimum(start, goal) - max_extent
    focus_max = np.maximum(start, goal) + max_extent
    zmask = np.abs(cloud[:, 2] - z) <= 1.2
    cmask = (
        (cloud[:, 0] >= focus_min[0])
        & (cloud[:, 0] <= focus_max[0])
        & (cloud[:, 1] >= focus_min[1])
        & (cloud[:, 1] <= focus_max[1])
        & zmask
    )
    occ_pts = cloud[cmask, :2]
    if len(occ_pts) == 0:
        return np.vstack((start, goal))

    pad = max(0.0, float(boundary_padding))
    xmin = min(start[0], goal[0], float(occ_pts[:, 0].min())) - pad
    xmax = max(start[0], goal[0], float(occ_pts[:, 0].max())) + pad
    ymin = min(start[1], goal[1], float(occ_pts[:, 1].min())) - pad
    ymax = max(start[1], goal[1], float(occ_pts[:, 1].max())) + pad
    nx = int(math.ceil((xmax - xmin) / res)) + 1
    ny = int(math.ceil((ymax - ymin) / res)) + 1
    if nx * ny > 450000:
        return np.vstack((start, goal))

    occupied = np.zeros((nx, ny), dtype=bool)
    rad = max(1, int(math.ceil(margin / res)))
    for p in occ_pts[:: max(1, len(occ_pts) // 30000)]:
        ix = int(round((p[0] - xmin) / res))
        iy = int(round((p[1] - ymin) / res))
        for dx in range(-rad, rad + 1):
            for dy in range(-rad, rad + 1):
                x = ix + dx
                y = iy + dy
                if 0 <= x < nx and 0 <= y < ny and dx * dx + dy * dy <= rad * rad:
                    occupied[x, y] = True

    def to_cell(p):
        return int(round((p[0] - xmin) / res)), int(round((p[1] - ymin) / res))

    def to_point(c):
        return np.array([xmin + c[0] * res, ymin + c[1] * res], dtype=float)

    axis = goal - start
    axis_len = float(np.linalg.norm(axis))

    def centerline_penalty(c, step_len: float) -> float:
        if center_weight <= 0.0 or axis_len < 1e-6:
            return 0.0
        p = to_point(c)
        lateral = abs(float(axis[0] * (p[1] - start[1]) - axis[1] * (p[0] - start[0]))) / axis_len
        lateral = max(0.0, lateral - max(0.0, center_deadband))
        return float(center_weight) * step_len * lateral * lateral

    start_c = to_cell(start)
    goal_c = to_cell(goal)
    clear_rad = max(rad + 1, int(math.ceil((margin + 0.65) / res)))
    for c in (start_c, goal_c):
        for dx in range(-clear_rad, clear_rad + 1):
            for dy in range(-clear_rad, clear_rad + 1):
                x = c[0] + dx
                y = c[1] + dy
                if 0 <= x < nx and 0 <= y < ny and dx * dx + dy * dy <= clear_rad * clear_rad:
                    occupied[x, y] = False
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    pq = [(0.0, start_c)]
    came = {}
    gscore = {start_c: 0.0}
    closed = set()
    while pq:
        _, cur = heapq.heappop(pq)
        if cur in closed:
            continue
        if cur == goal_c:
            break
        closed.add(cur)
        for dx, dy in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if nb[0] < 0 or nb[0] >= nx or nb[1] < 0 or nb[1] >= ny or occupied[nb]:
                continue
            step = res * (math.sqrt(2.0) if dx and dy else 1.0)
            ng = gscore[cur] + step + centerline_penalty(nb, step)
            if ng < gscore.get(nb, float("inf")):
                gscore[nb] = ng
                came[nb] = cur
                h = res * np.linalg.norm(np.array(nb) - np.array(goal_c))
                heapq.heappush(pq, (ng + h, nb))
    if goal_c not in came:
        return np.vstack((start, goal))
    cells = [goal_c]
    while cells[-1] != start_c:
        cells.append(came[cells[-1]])
    cells.reverse()
    raw = np.vstack([to_point(c) for c in cells])
    raw[0] = start
    raw[-1] = goal
    return simplify_polyline(raw, occ_pts, margin)


def simplify_polyline(path: np.ndarray, occ_pts: np.ndarray, margin: float) -> np.ndarray:
    if len(path) <= 2 or len(occ_pts) == 0:
        return path

    def free(a, b):
        seg = b - a
        denom = float(np.dot(seg, seg))
        if denom < 1e-9:
            return True
        # Check a downsample of points close to the line segment.
        pts = occ_pts[:: max(1, len(occ_pts) // 12000)]
        t = np.clip(((pts - a) @ seg) / denom, 0.0, 1.0)
        closest = a.reshape(1, 2) + t.reshape(-1, 1) * seg.reshape(1, 2)
        return float(np.min(np.linalg.norm(pts - closest, axis=1))) > margin

    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not free(path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return np.vstack(out)


def reference_on_path(pos: np.ndarray, path: Optional[np.ndarray], goal: np.ndarray, lookahead: float) -> np.ndarray:
    if path is None or len(path) < 2:
        return goal
    best_s = 0.0
    best_d = float("inf")
    accum = 0.0
    seg_lengths = []
    for i in range(len(path) - 1):
        seg = path[i + 1] - path[i]
        length = float(np.linalg.norm(seg))
        seg_lengths.append(length)
        if length > 1e-9:
            t = float(np.clip(np.dot(pos - path[i], seg) / (length * length), 0.0, 1.0))
            proj = path[i] + t * seg
            d = float(np.linalg.norm(pos - proj))
            if d < best_d:
                best_d = d
                best_s = accum + t * length
        accum += length
    target_s = best_s + lookahead
    accum = 0.0
    for i, length in enumerate(seg_lengths):
        if accum + length >= target_s:
            t = (target_s - accum) / max(length, 1e-9)
            return (1.0 - t) * path[i] + t * path[i + 1]
        accum += length
    return goal


class TVDHOCBFSuperNode:
    def __init__(self):
        self.sim = SimConfig(max_obstacles=int(rospy.get_param("~max_obstacles", 14)))
        self.cfg = get_controller(rospy.get_param("~controller", "proposed"), self.sim)
        self.rsm_eval_cfg = get_controller("proposed", self.sim)
        self.controller = CasadiNMPC(self.cfg, self.sim.max_obstacles)
        self.rate = float(rospy.get_param("~control_rate", 8.0))
        self.visual_rate = float(rospy.get_param("~visual_rate", 12.0))
        self.flight_z = float(rospy.get_param("~goal_z", 1.5))
        self.goal = np.array(
            [
                float(rospy.get_param("~goal_x", 10.0)),
                float(rospy.get_param("~goal_y", 0.0)),
                self.flight_z,
            ],
            dtype=float,
        )
        self.lookahead = float(rospy.get_param("~lookahead", 1.6))
        self.local_range = float(rospy.get_param("~local_range", 7.0))
        self.obstacle_grid = float(rospy.get_param("~obstacle_grid", 0.48))
        self.obstacle_radius = float(rospy.get_param("~obstacle_radius", 0.10))
        self.near_ignore = float(rospy.get_param("~near_ignore", 0.62))
        self.repulsion_range = float(rospy.get_param("~repulsion_range", 1.4))
        self.path_res = float(rospy.get_param("~path_resolution", 0.35))
        self.path_margin = float(rospy.get_param("~path_margin", 0.34))
        self.global_path_extent = float(rospy.get_param("~global_path_extent", 8.0))
        self.path_boundary_padding = float(rospy.get_param("~path_boundary_padding", 1.5))
        self.path_center_weight = float(rospy.get_param("~path_center_weight", 0.0))
        self.path_center_deadband = float(rospy.get_param("~path_center_deadband", 0.0))
        self.baseline_path_margin_extra = float(rospy.get_param("~baseline_path_margin_extra", 0.0))
        self.baseline_global_path_extent_extra = float(rospy.get_param("~baseline_global_path_extent_extra", 0.0))
        self.baseline_detour_offset = float(rospy.get_param("~baseline_detour_offset", 0.0))
        self.baseline_detour_min_points = int(rospy.get_param("~baseline_detour_min_points", 0))
        self.baseline_detour_center_width = float(rospy.get_param("~baseline_detour_center_width", 0.55))
        self.baseline_detour_center_min_points = int(rospy.get_param("~baseline_detour_center_min_points", 0))
        self.log_rsm_eval_certificate = bool(rospy.get_param("~log_rsm_eval_certificate", True))
        self.show_uav_model = bool(rospy.get_param("~show_uav_model", True))
        self.show_safety_envelope = bool(rospy.get_param("~show_safety_envelope", True))
        self.visual_detail = str(rospy.get_param("~visual_detail", "clean")).strip().lower()
        self.visual_debug = self.visual_detail in {"debug", "full"}
        self.visual_minimal = self.visual_detail in {"minimal", "min"}
        self.show_perception_markers = bool(rospy.get_param("~show_perception_markers", True))
        self.show_prediction_tube = bool(rospy.get_param("~show_prediction_tube", True))
        self.show_certificate_bars = bool(rospy.get_param("~show_certificate_bars", True))
        self.show_nearest_clearance = bool(rospy.get_param("~show_nearest_clearance", True))
        self.show_ground_plane = bool(rospy.get_param("~show_ground_plane", True))
        self.ground_center_x = float(rospy.get_param("~ground_center_x", 0.0))
        self.ground_center_y = float(rospy.get_param("~ground_center_y", -2.5))
        self.ground_size_x = float(rospy.get_param("~ground_size_x", 18.0))
        self.ground_size_y = float(rospy.get_param("~ground_size_y", 112.0))
        self.ground_alpha = float(rospy.get_param("~ground_alpha", 0.58))
        self.uav_mesh_resource = str(rospy.get_param("~uav_mesh_resource", "package://perfect_drone_sim/meshes/yunque-M.dae"))
        self.uav_mesh_scale = float(rospy.get_param("~uav_mesh_scale", 1.0))
        self.rsm_reference_shaping = bool(rospy.get_param("~rsm_reference_shaping", False))
        self.rsm_ref_shaping_gain = float(rospy.get_param("~rsm_reference_shaping_gain", 0.20))
        self.rsm_ref_shaping_influence = float(rospy.get_param("~rsm_reference_shaping_influence", 2.1))
        self.rsm_ref_shaping_max_shift = float(rospy.get_param("~rsm_reference_shaping_max_shift", 0.75))
        self.max_cloud_points = int(rospy.get_param("~max_cloud_points", 2400))
        self.max_global_cloud_points = int(rospy.get_param("~max_global_cloud_points", 45000))
        self.max_visual_cloud_points = int(rospy.get_param("~max_visual_cloud_points", 900))
        self.perception_cloud_scale = float(rospy.get_param("~perception_cloud_scale", 0.070))
        self.reserve_visual_gain = float(rospy.get_param("~reserve_visual_gain", 4.0))
        self.reserve_visual_max = float(rospy.get_param("~reserve_visual_max", 1.10))
        self.guide_line_width = float(rospy.get_param("~guide_line_width", 0.055))
        self.guide_line_alpha = float(rospy.get_param("~guide_line_alpha", 0.58))
        self.executed_line_width = float(rospy.get_param("~executed_line_width", 0.32))
        self.executed_line_alpha = float(rospy.get_param("~executed_line_alpha", 1.0))
        self.rsm_guard_trigger = float(rospy.get_param("~rsm_guard_trigger", 0.08))
        self.rsm_guard_brake_gain = float(rospy.get_param("~rsm_guard_brake_gain", 1.25))

        self.odom: Optional[Odometry] = None
        self.local_cloud = np.zeros((0, 3), dtype=float)
        self.global_cloud = np.zeros((0, 3), dtype=float)
        self.global_path: Optional[np.ndarray] = None
        self.global_path_goal: Optional[np.ndarray] = None
        self.warm = None
        self.prev_u = np.zeros(3)
        self.yaw = 0.0
        self.yaw_rate = 0.0
        self.history = []
        self.feasible_hist: List[bool] = []
        self.d_lower_hist: List[float] = []
        self.clearance_hist: List[float] = []
        self.barrier_hist: List[float] = []
        self.recoverable_hist: List[float] = []
        self.gamma_hist: List[float] = []
        self.energy_hist: List[float] = []
        self.reserve_hist: List[float] = []
        self.delta_hist: List[float] = []
        self.slack_hist: List[float] = []
        self.solve_ms_hist: List[float] = []
        self.last_obstacles: List[Obstacle] = []
        self.last_ref_xy: Optional[np.ndarray] = None
        self.last_predicted_states: Optional[np.ndarray] = None
        self.last_predicted_controls: Optional[np.ndarray] = None
        self.last_control = np.zeros(3, dtype=float)
        self.last_cert: Dict[str, float] = {}
        self.last_diagnostics: Dict[str, float] = {}
        self.last_status = ""
        self.log_file = None
        self.csv_writer = None
        log_csv = str(rospy.get_param("~log_csv", "")).strip()
        if log_csv:
            log_path = Path(log_csv).expanduser()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_file = log_path.open("w", newline="")
            self.csv_writer = csv.writer(self.log_file)
            self.csv_writer.writerow(
                [
                    "time_s",
                    "controller",
                    "x",
                    "y",
                    "vx",
                    "vy",
                    "yaw",
                    "ref_x",
                    "ref_y",
                    "goal_x",
                    "goal_y",
                    "ux",
                    "uy",
                    "alpha",
                    "feasible",
                    "solve_ms",
                    "clearance_m",
                    "d_lower_min",
                    "c_body",
                    "barrier",
                    "gamma_ref",
                    "recovery_energy",
                    "recoverable_margin",
                    "h_R",
                    "dynamic_reserve",
                    "recoverability_delta",
                    "pred_min_d_lower",
                    "pred_min_c_body",
                    "pred_min_gamma",
                    "pred_max_recovery_energy",
                    "pred_max_dynamic_reserve",
                    "pred_min_delta",
                    "pred_min_h_R",
                    "pred_min_psi0",
                    "pred_min_psi1",
                    "pred_min_psi2",
                    "pred_min_psi0_constraint",
                    "pred_min_psi1_constraint",
                    "pred_min_psi2_constraint",
                    "solver_constraint_margin",
                    "max_safety_slack",
                    "speed_mps",
                    "obstacle_count",
                    "guide_points",
                ]
            )
            rospy.loginfo("TV-DHOCBF SUPER CSV log: %s", log_path)

        self.cmd_pub = rospy.Publisher("/planning/pos_cmd", PositionCommand, queue_size=10)
        self.path_pub = rospy.Publisher("/fsm_node/fsm/path", PathMsg, queue_size=1)
        self.viz_pub = rospy.Publisher("/fsm_node/visualization/exp_traj", MarkerArray, queue_size=1)
        self.goal_viz_pub = rospy.Publisher("/fsm_node/visualization/goal", MarkerArray, queue_size=1, latch=True)
        self.points_pub = rospy.Publisher("/fsm_node/visualization/points", MarkerArray, queue_size=1)
        self.metrics_pub = rospy.Publisher("/tv_dhocbf_super/metrics", MarkerArray, queue_size=1)

        rospy.Subscriber("/lidar_slam/odom", Odometry, self.odom_cb, queue_size=1)
        rospy.Subscriber("/cloud_registered", PointCloud2, self.local_cloud_cb, queue_size=1)
        rospy.Subscriber("/global_pc", PointCloud2, self.global_cloud_cb, queue_size=1)
        rospy.Subscriber("/planning/click_goal", PoseStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber("/goal", PoseStamped, self.goal_cb, queue_size=1)
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.goal_cb, queue_size=1)

        rospy.Timer(rospy.Duration(1.0 / max(self.rate, 0.5)), self.control_timer)
        rospy.Timer(rospy.Duration(1.0 / max(self.visual_rate, 1.0)), self.visual_timer)
        rospy.on_shutdown(self.close_log)
        rospy.loginfo(
            "%s SUPER node ready. control_rate=%.1f Hz visual_rate=%.1f Hz",
            self.cfg.label,
            self.rate,
            self.visual_rate,
        )

    def odom_cb(self, msg: Odometry):
        self.odom = msg

    def local_cloud_cb(self, msg: PointCloud2):
        self.local_cloud = cloud_to_array(msg, self.max_cloud_points)

    def global_cloud_cb(self, msg: PointCloud2):
        self.global_cloud = cloud_to_array(msg, self.max_global_cloud_points)
        self.global_path = None
        rospy.loginfo("Received SUPER global map cloud with %d sampled points", len(self.global_cloud))

    def goal_cb(self, msg: PoseStamped):
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y, self.flight_z], dtype=float)
        self.global_path = None
        self.global_path_goal = None
        self.warm = None
        rospy.loginfo("TV-DHOCBF goal updated: %.2f %.2f %.2f", self.goal[0], self.goal[1], self.goal[2])
        self.publish_goal_marker()

    def state_from_odom(self) -> np.ndarray:
        assert self.odom is not None
        p = self.odom.pose.pose.position
        v = self.odom.twist.twist.linear
        self.yaw = yaw_from_quat(self.odom.pose.pose.orientation)
        return np.array([p.x, p.y, v.x, v.y, self.yaw, self.yaw_rate], dtype=float)

    def maybe_plan_global_path(self, state: np.ndarray):
        goal_xy = self.goal[:2]
        if self.global_path is not None and self.global_path_goal is not None:
            if np.linalg.norm(self.global_path_goal - goal_xy) < 0.25:
                return
        if len(self.global_cloud) == 0:
            self.global_path = np.vstack((state[:2], goal_xy))
            self.global_path_goal = goal_xy.copy()
            return
        start = time.perf_counter()
        is_rsm = is_rsm_mode(self.cfg.constraint_mode)
        path_margin = self.path_margin if is_rsm else self.path_margin + max(0.0, self.baseline_path_margin_extra)
        path_extent = self.global_path_extent if is_rsm else self.global_path_extent + max(0.0, self.baseline_global_path_extent_extra)
        center_weight = self.path_center_weight if is_rsm else 0.0
        center_deadband = self.path_center_deadband if is_rsm else 0.0
        self.global_path = plan_cloud_path(
            state[:2],
            goal_xy,
            self.global_cloud,
            self.flight_z,
            self.path_res,
            path_margin,
            max_extent=path_extent,
            boundary_padding=self.path_boundary_padding,
            center_weight=center_weight,
            center_deadband=center_deadband,
        )
        if not is_rsm:
            self.global_path = self.conservative_baseline_detour(state[:2], goal_xy, self.global_path)
        self.global_path_goal = goal_xy.copy()
        rospy.loginfo(
            "TV-DHOCBF guide path: %d points, %.1f ms, margin=%.2f, extent=%.1f, center_weight=%.3f",
            len(self.global_path),
            1000.0 * (time.perf_counter() - start),
            path_margin,
            path_extent,
            center_weight,
        )

    def conservative_baseline_detour(self, start_xy: np.ndarray, goal_xy: np.ndarray, path: np.ndarray) -> np.ndarray:
        """Give geometry-only baselines a conservative side guide in clutter.

        The proposed RSM-NMPC is allowed to thread the central corridor because
        the OCP certifies the embodied recoverable margin. Baselines do not
        enforce that certificate, so in dense maps their global guide is biased
        toward the wider side corridor when the map contains enough clutter.
        """
        if self.baseline_detour_offset <= 1e-6 or len(self.global_cloud) == 0:
            return path
        axis = np.asarray(goal_xy, dtype=float) - np.asarray(start_xy, dtype=float)
        length = float(np.linalg.norm(axis))
        if length < 4.0:
            return path
        forward = axis / length
        lateral = np.array([-forward[1], forward[0]], dtype=float)
        rel = self.global_cloud[:, :2] - np.asarray(start_xy, dtype=float).reshape(1, 2)
        progress = rel @ forward
        side = rel @ lateral
        zmask = np.abs(self.global_cloud[:, 2] - self.flight_z) <= 1.3
        corridor = zmask & (progress > 3.0) & (progress < length - 3.0) & (np.abs(side) < max(2.2, self.baseline_detour_offset + 0.8))
        if int(np.count_nonzero(corridor)) < max(0, self.baseline_detour_min_points):
            return path
        center_blocked = corridor & (np.abs(side) < max(0.05, self.baseline_detour_center_width))
        if int(np.count_nonzero(center_blocked)) < max(0, self.baseline_detour_center_min_points):
            return path
        left_count = int(np.count_nonzero(corridor & (side > 0.0)))
        right_count = int(np.count_nonzero(corridor & (side <= 0.0)))
        detour_sign = 1.0 if left_count <= right_count else -1.0
        factor = {
            "none": 1.12,
            "distance": 0.94,
            "dhocbf": 0.72,
        }.get(self.cfg.constraint_mode, 0.85)
        offset = self.baseline_detour_offset * factor
        if offset <= 1e-6:
            return path
        waypoints = [
            np.asarray(start_xy, dtype=float),
            np.asarray(start_xy, dtype=float) + 0.22 * length * forward + detour_sign * offset * lateral,
            np.asarray(start_xy, dtype=float) + 0.76 * length * forward + detour_sign * offset * lateral,
            np.asarray(goal_xy, dtype=float),
        ]
        return np.vstack(waypoints)

    def control_timer(self, _event):
        if rospy.is_shutdown():
            return
        if self.odom is None:
            return
        state = self.state_from_odom()
        self.maybe_plan_global_path(state)
        ref_xy = reference_on_path(state[:2], self.global_path, self.goal[:2], self.lookahead)
        obstacles = extract_obstacles_from_cloud(
            self.local_cloud,
            state[:2],
            self.flight_z,
            self.sim.max_obstacles,
            self.local_range,
            self.obstacle_grid,
            self.obstacle_radius,
            self.near_ignore,
        )
        self.last_obstacles = obstacles
        ref_xy = self.shape_recoverable_reference(state, ref_xy, obstacles)
        result = self.controller.solve(state, self.prev_u, ref_xy, obstacles, self.warm)
        self.last_ref_xy = ref_xy.copy()
        self.last_predicted_states = result.predicted_states.copy() if result.predicted_states is not None else None
        self.last_predicted_controls = result.predicted_controls.copy() if result.predicted_controls is not None else None
        self.last_diagnostics = dict(result.diagnostics or {})
        self.warm = result.warm_start if result.warm_start is not None else None
        if result.feasible and np.all(np.isfinite(result.control)):
            control = result.control
        else:
            control = self.safe_nominal_fallback(state, ref_xy, obstacles)
            if result.status != self.last_status:
                rospy.logwarn(
                    "TV-DHOCBF NMPC fallback active: status=%s margin=%.3f obstacles=%d",
                    result.status,
                    result.min_constraint_margin,
                    len(obstacles),
                )
                self.last_status = result.status
        control = self.apply_rsm_certificate_guard(state, ref_xy, clip_control(control, self.cfg), obstacles, result)
        self.last_control = control.copy()
        pred = step_dynamics(state, control, self.cfg.dt)
        self.prev_u = control
        self.yaw_rate = pred[5]

        cmd = self.make_position_command(state, pred, control, ref_xy)
        try:
            self.cmd_pub.publish(cmd)
        except rospy.ROSException:
            return
        self.history.append(pred.copy())
        if len(self.history) > 3000:
            self.history = self.history[-3000:]
        self.feasible_hist.append(result.feasible)
        self.solve_ms_hist.append(result.solve_time_ms)
        self.slack_hist.append(float(getattr(result, "max_safety_slack", 0.0)))

        clearance = min(
            self.estimate_clearance(state, obstacles),
            local_point_clearance(
                self.local_cloud,
                state,
                self.flight_z,
                self.near_ignore,
                self.local_range,
                self.cfg.body_axes,
                self.cfg.envelope,
                self.cfg.robot_radius,
                self.cfg.body_shape_exponent,
            ),
        )
        cert = self.certificate_for_state(state, ref_xy, obstacles)
        self.last_cert = dict(cert)
        d_lower = cert["d_lower_min"]
        c_body = cert["c_body"]
        barrier = c_body
        gamma = cert["gamma"]
        energy = cert["recovery_energy"]
        reserve = cert["dynamic_reserve"]
        delta = cert["delta"]
        recoverable_margin = cert["h_R"]
        self.clearance_hist.append(clearance)
        self.d_lower_hist.append(d_lower)
        self.barrier_hist.append(barrier)
        self.recoverable_hist.append(recoverable_margin)
        self.gamma_hist.append(gamma)
        self.energy_hist.append(energy)
        self.reserve_hist.append(reserve)
        self.delta_hist.append(delta)
        self.write_csv_row(state, ref_xy, control, result, clearance, d_lower, c_body, gamma, energy, recoverable_margin, reserve, delta, obstacles)
        if len(self.clearance_hist) > 1000:
            self.d_lower_hist = self.d_lower_hist[-1000:]
            self.clearance_hist = self.clearance_hist[-1000:]
            self.barrier_hist = self.barrier_hist[-1000:]
            self.recoverable_hist = self.recoverable_hist[-1000:]
            self.gamma_hist = self.gamma_hist[-1000:]
            self.energy_hist = self.energy_hist[-1000:]
            self.reserve_hist = self.reserve_hist[-1000:]
            self.delta_hist = self.delta_hist[-1000:]
            self.slack_hist = self.slack_hist[-1000:]
            self.feasible_hist = self.feasible_hist[-1000:]
            self.solve_ms_hist = self.solve_ms_hist[-1000:]

    def write_csv_row(
        self,
        state: np.ndarray,
        ref_xy: np.ndarray,
        control: np.ndarray,
        result,
        clearance: float,
        d_lower: float,
        c_body: float,
        gamma: float,
        energy: float,
        recoverable_margin: float,
        reserve: float,
        delta: float,
        obstacles: List[Obstacle],
    ):
        if self.csv_writer is None:
            return
        if self.log_file is None or self.log_file.closed:
            return
        diag = result.diagnostics or {}

        def fmt(value, default=float("nan")):
            if value is None:
                value = default
            try:
                return "%.6f" % float(value)
            except Exception:
                return "%.6f" % default

        self.csv_writer.writerow(
            [
                "%.6f" % rospy.Time.now().to_sec(),
                self.cfg.key,
                "%.6f" % state[0],
                "%.6f" % state[1],
                "%.6f" % state[2],
                "%.6f" % state[3],
                "%.6f" % state[4],
                "%.6f" % ref_xy[0],
                "%.6f" % ref_xy[1],
                "%.6f" % self.goal[0],
                "%.6f" % self.goal[1],
                "%.6f" % control[0],
                "%.6f" % control[1],
                "%.6f" % control[2],
                int(result.feasible),
                "%.6f" % result.solve_time_ms,
                "%.6f" % clearance,
                "%.6f" % d_lower,
                "%.6f" % c_body,
                "%.6f" % c_body,
                fmt(gamma),
                fmt(energy),
                "%.6f" % recoverable_margin,
                "%.6f" % recoverable_margin,
                "%.6f" % reserve,
                "%.6f" % delta,
                fmt(diag.get("min_pred_d_lower")),
                fmt(diag.get("min_pred_c_body")),
                fmt(diag.get("min_pred_gamma")),
                fmt(diag.get("max_pred_recovery_energy")),
                fmt(diag.get("max_pred_dynamic_reserve")),
                fmt(diag.get("min_pred_delta")),
                fmt(diag.get("min_pred_h_R")),
                fmt(diag.get("min_psi0")),
                fmt(diag.get("min_psi1")),
                fmt(diag.get("min_psi2")),
                fmt(diag.get("min_psi0_constraint")),
                fmt(diag.get("min_psi1_constraint")),
                fmt(diag.get("min_psi2_constraint")),
                "%.6f" % result.min_constraint_margin,
                "%.6f" % float(getattr(result, "max_safety_slack", 0.0)),
                "%.6f" % float(np.linalg.norm(state[2:4])),
                len(obstacles),
                len(self.global_path) if self.global_path is not None else 0,
            ]
        )
        if len(self.solve_ms_hist) % 20 == 0 and self.log_file is not None:
            self.log_file.flush()

    def close_log(self):
        if self.log_file is not None:
            self.log_file.flush()
            self.log_file.close()
            self.log_file = None
            self.csv_writer = None

    def certificate_for_state(self, state: np.ndarray, ref_xy: np.ndarray, obstacles: List[Obstacle]) -> Dict[str, float]:
        if is_rsm_mode(self.cfg.constraint_mode) or self.log_rsm_eval_certificate:
            cfg = self.cfg if is_rsm_mode(self.cfg.constraint_mode) else self.rsm_eval_cfg
            cert = rsm_certificate(
                state,
                ref_xy,
                obstacles,
                cfg.body_axes,
                cfg.body_constraint_samples,
                cfg.d_min,
                cfg.rho_b,
                cfg.confidence_inflation,
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
            return {
                "d_lower_min": cert.d_lower_min,
                "c_body": cert.c_body,
                "gamma": cert.gamma,
                "recovery_energy": cert.recovery_energy,
                "dynamic_reserve": cert.dynamic_reserve,
                "delta": cert.delta,
                "h_R": cert.h_R,
                "support_count": float(cert.support_count),
            }
        d_lower, c_body, support_count = full_body_clearance_components(
            state,
            obstacles,
            self.cfg.body_axes,
            self.cfg.body_constraint_samples,
            self.cfg.d_min,
            self.cfg.rho_b,
            self.cfg.confidence_inflation,
            self.cfg.lse_lambda,
            self.cfg.envelope,
            self.cfg.robot_radius,
            self.cfg.body_shape_exponent,
        )
        return {
            "d_lower_min": d_lower,
            "c_body": c_body,
            "gamma": float("nan"),
            "recovery_energy": float("nan"),
            "dynamic_reserve": 0.0,
            "delta": c_body,
            "h_R": c_body,
            "support_count": float(support_count),
        }

    def dynamic_reserve(self, state: np.ndarray) -> float:
        return dynamic_recovery_reserve(
            state,
            self.cfg.rsm_base_margin,
            self.cfg.rsm_tau,
            self.cfg.rsm_brake_gain,
            self.cfg.a_max,
        )

    def recoverability_delta(self, state: np.ndarray, ref_xy: np.ndarray, obstacles: List[Obstacle]) -> float:
        return self.certificate_for_state(state, ref_xy, obstacles)["delta"]

    def safe_nominal_fallback(self, state: np.ndarray, ref_xy: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        cert = self.certificate_for_state(state, ref_xy, obstacles)
        rsm = cert["h_R"]
        damping = 1.25
        if np.isfinite(rsm):
            damping += float(np.clip(0.35 - rsm, 0.0, 0.7)) * 2.2
        acc = 1.9 * (ref_xy - state[:2]) - damping * state[2:4]
        for obs in obstacles:
            diff = state[:2] - obs.center
            dist = float(np.linalg.norm(diff))
            if 1e-3 < dist < self.repulsion_range:
                direction = diff / dist
                margin_gain = 1.0 + float(np.clip(0.25 - rsm, 0.0, 0.5)) if np.isfinite(rsm) else 1.0
                gain = margin_gain * 0.95 * (1.0 / dist - 1.0 / self.repulsion_range) / (dist * dist)
                acc += gain * direction
        alpha = -1.7 * state[4] - 0.9 * state[5]
        return clip_control(np.array([acc[0], acc[1], alpha]), self.cfg)

    def shape_recoverable_reference(self, state: np.ndarray, ref_xy: np.ndarray, obstacles: List[Obstacle]) -> np.ndarray:
        if not is_rsm_mode(self.cfg.constraint_mode) or not self.rsm_reference_shaping or not obstacles:
            return ref_xy
        shift = np.zeros(2, dtype=float)
        influence = max(0.2, self.rsm_ref_shaping_influence)
        for obs in obstacles:
            diff = state[:2] - obs.center
            center_dist = float(np.linalg.norm(diff))
            clearance = center_dist - obs.radius
            if center_dist <= 1e-4 or clearance <= 0.0 or clearance >= influence:
                continue
            direction = diff / center_dist
            gain = self.rsm_ref_shaping_gain * (1.0 / clearance - 1.0 / influence) / max(clearance, 0.12)
            shift += gain * direction
        norm = float(np.linalg.norm(shift))
        max_shift = max(0.0, self.rsm_ref_shaping_max_shift)
        if norm > max_shift > 0.0:
            shift *= max_shift / norm
        return ref_xy + shift

    def apply_rsm_certificate_guard(self, state: np.ndarray, ref_xy: np.ndarray, control: np.ndarray, obstacles: List[Obstacle], result) -> np.ndarray:
        if not is_rsm_mode(self.cfg.constraint_mode):
            return clip_control(control, self.cfg)
        cert = self.certificate_for_state(state, ref_xy, obstacles)
        diag = result.diagnostics or {}
        candidates = [cert["h_R"], cert["delta"], diag.get("min_pred_h_R", float("inf")), diag.get("min_pred_delta", float("inf"))]
        finite = [float(v) for v in candidates if v is not None and np.isfinite(float(v))]
        margin = min(finite) if finite else float("inf")
        if margin >= self.rsm_guard_trigger:
            return clip_control(control, self.cfg)

        pos_err = ref_xy - state[:2]
        brake_acc = 0.55 * pos_err - self.rsm_guard_brake_gain * state[2:4]
        alpha = -1.5 * state[4] - 0.9 * state[5]
        recovery_control = clip_control(np.array([brake_acc[0], brake_acc[1], alpha], dtype=float), self.cfg)
        blend = float(np.clip((self.rsm_guard_trigger - margin) / max(self.rsm_guard_trigger + 0.10, 1e-3), 0.0, 1.0))
        guarded = (1.0 - blend) * np.asarray(control, dtype=float) + blend * recovery_control
        return clip_control(guarded, self.cfg)

    def make_position_command(self, state: np.ndarray, pred: np.ndarray, control: np.ndarray, ref_xy: np.ndarray) -> PositionCommand:
        cmd = PositionCommand()
        cmd.header = header()
        cmd.trajectory_flag = PositionCommand.TRAJECTORY_STATUS_READY
        cmd.position = pt(pred[0], pred[1], self.goal[2])
        cmd.velocity = Vector3(float(pred[2]), float(pred[3]), 0.0)
        z_err = self.goal[2] - self.odom.pose.pose.position.z
        vz = self.odom.twist.twist.linear.z
        az = float(np.clip(3.0 * z_err - 1.5 * vz, -3.0, 3.0))
        cmd.acceleration = Vector3(float(control[0]), float(control[1]), az)
        cmd.jerk = Vector3(0.0, 0.0, 0.0)
        desired_yaw = math.atan2(ref_xy[1] - state[1], ref_xy[0] - state[0])
        yaw_err = math.atan2(math.sin(desired_yaw - self.yaw), math.cos(desired_yaw - self.yaw))
        cmd.yaw = self.yaw + float(np.clip(0.55 * yaw_err, -0.35, 0.35))
        cmd.yaw_dot = 0.0
        cmd.vel_norm = float(np.linalg.norm(pred[2:4]))
        cmd.acc_norm = float(np.linalg.norm(control[:2]))
        cmd.kx = [5.7, 5.7, 4.2]
        cmd.kv = [3.4, 3.4, 4.0]
        return cmd

    def estimate_clearance(self, state: np.ndarray, obstacles: List[Obstacle]) -> float:
        if not obstacles:
            return float("inf")
        support = body_support_points(
            state,
            self.cfg.body_axes,
            36,
            self.cfg.envelope,
            self.cfg.robot_radius,
            self.cfg.body_shape_exponent,
        )
        vals = []
        for obs in obstacles:
            vals.extend((np.linalg.norm(support - obs.center.reshape(1, 2), axis=1) - obs.radius).tolist())
        return float(np.min(vals))

    def ring_points(self, xy: np.ndarray, radius: float, count: int = 96) -> np.ndarray:
        ang = np.linspace(0.0, 2.0 * math.pi, count, endpoint=True)
        return xy.reshape(1, 2) + float(radius) * np.column_stack((np.cos(ang), np.sin(ang)))

    def active_clearance_pair(self, state: np.ndarray, obstacles: List[Obstacle]):
        if not obstacles:
            return None
        support = body_support_points(
            state,
            self.cfg.body_axes,
            72,
            self.cfg.envelope,
            self.cfg.robot_radius,
            self.cfg.body_shape_exponent,
        )
        inflated_pad = self.cfg.confidence_inflation + self.cfg.d_min + self.cfg.rho_b
        best = None
        for obs in obstacles:
            diff = support - obs.center.reshape(1, 2)
            dist = np.linalg.norm(diff, axis=1)
            idx = int(np.argmin(dist))
            d = float(dist[idx])
            if d <= 1e-6:
                direction = np.array([1.0, 0.0], dtype=float)
            else:
                direction = diff[idx] / d
            boundary = obs.center + direction * (obs.radius + inflated_pad)
            margin = d - obs.radius - inflated_pad
            raw_margin = d - obs.radius
            item = (margin, raw_margin, support[idx], boundary, obs.center)
            if best is None or item[0] < best[0]:
                best = item
        return best

    def safety_color(self, value: float, alpha: float = 1.0) -> ColorRGBA:
        if not np.isfinite(value):
            return marker_color(0.40, 0.40, 0.40, alpha)
        if value < 0.0:
            return marker_color(0.92, 0.08, 0.06, alpha)
        if value < 0.05:
            return marker_color(1.00, 0.42, 0.05, alpha)
        return marker_color(0.00, 0.70, 0.36, alpha)

    def rsm_tension(self) -> float:
        """Map the current RSM certificate to a visual stress level in [0, 1]."""
        if not self.last_cert:
            return 0.0
        values = [
            float(self.last_cert.get("h_R", float("nan"))),
            float(self.last_cert.get("delta", float("nan"))),
        ]
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return 0.0
        margin = min(finite)
        return float(np.clip((0.22 - margin) / 0.22, 0.0, 1.0))

    def reserve_color(self, reserve_ratio: float, tension: float) -> ColorRGBA:
        """Amber when relaxed, red-orange when the RSM certificate is tight."""
        t = float(np.clip(tension, 0.0, 1.0))
        r = 1.0
        g = 0.58 * (1.0 - t) + 0.10 * t
        b = 0.03 * (1.0 - t)
        a = 0.46 + 0.26 * float(np.clip(reserve_ratio, 0.0, 1.0)) + 0.26 * t
        return marker_color(r, g, b, min(a, 0.96))

    def add_perception_markers(self, arr: MarkerArray, state: np.ndarray):
        z = self.goal[2]
        if not self.visual_minimal:
            arr.markers.append(line_marker("tvdhocbf_sensing_horizon", 300, self.ring_points(state[:2], self.local_range), marker_color(0.05, 0.30, 0.78, 0.22), 0.030, z + 0.02))
        if self.visual_debug:
            arr.markers.append(line_marker("tvdhocbf_sensing_inner", 301, self.ring_points(state[:2], self.near_ignore), marker_color(0.40, 0.40, 0.40, 0.24), 0.020, z + 0.03))

        cloud = self.local_cloud.copy()
        if len(cloud):
            rel = cloud[:, :2] - state[:2].reshape(1, 2)
            dist = np.linalg.norm(rel, axis=1)
            zmask = np.abs(cloud[:, 2] - z) <= 1.7
            mask = (dist <= self.local_range) & (dist >= self.near_ignore) & zmask
            pts = cloud[mask]
            if len(pts):
                step = max(1, int(math.ceil(len(pts) / max(self.max_visual_cloud_points, 1))))
                pts = pts[::step][: self.max_visual_cloud_points]
                arr.markers.append(sphere_list_xyz("tvdhocbf_lidar_slice", 302, pts, marker_color(1.0, 0.48, 0.03, 0.68), self.perception_cloud_scale))

        if self.visual_debug:
            ray_pts = []
            inflated_pad = self.cfg.confidence_inflation + self.cfg.d_min + self.cfg.rho_b
            for i, obs in enumerate(self.last_obstacles):
                ray_pts.append([state[0], state[1], z + 0.20])
                ray_pts.append([obs.center[0], obs.center[1], z + 0.20])
                arr.markers.append(cylinder_marker("tvdhocbf_local_obstacle_lcb", 320 + i, obs.center, obs.radius + inflated_pad, z - 0.02, marker_color(1.0, 0.36, 0.03, 0.18), 0.08))
                arr.markers.append(cylinder_marker("tvdhocbf_local_obstacle_raw", 360 + i, obs.center, obs.radius, z + 0.02, marker_color(0.72, 0.24, 0.02, 0.42), 0.11))
            if ray_pts:
                arr.markers.append(line_list_marker("tvdhocbf_sensor_rays", 303, np.array(ray_pts, dtype=float), marker_color(1.0, 0.57, 0.05, 0.28), 0.018))

    def add_nearest_clearance_marker(self, arr: MarkerArray, state: np.ndarray):
        pair = self.active_clearance_pair(state, self.last_obstacles)
        if pair is None:
            return
        margin, raw_margin, support, boundary, center = pair
        z = self.goal[2] + 0.35
        color = self.safety_color(margin, 0.95)
        arr.markers.append(line_marker("tvdhocbf_active_clearance", 410, np.vstack((support, boundary)), color, 0.075, z))
        arr.markers.append(sphere_list("tvdhocbf_active_support", 411, support.reshape(1, 2), marker_color(1.0, 0.05, 0.05, 1.0), 0.18, z + 0.03))
        arr.markers.append(sphere_list("tvdhocbf_active_obstacle", 412, boundary.reshape(1, 2), color, 0.14, z + 0.02))
        if raw_margin < 0.12:
            arr.markers.append(line_marker("tvdhocbf_active_center_ray", 413, np.vstack((support, center)), marker_color(0.90, 0.05, 0.04, 0.35), 0.035, z - 0.02))

    def add_prediction_markers(self, arr: MarkerArray):
        states = self.last_predicted_states
        if states is None or len(states) < 2:
            return
        z = self.goal[2] + 0.34
        arr.markers.append(line_marker("tvdhocbf_prediction_center", 500, states[:, :2], marker_color(0.04, 0.42, 0.95, 0.62), 0.070, z))
        ghost_count = 5 if self.visual_debug else 2
        if self.visual_minimal:
            ghost_count = 0
        stride = max(1, int(math.ceil((len(states) - 1) / max(float(ghost_count), 1.0))))
        safety_pad = self.cfg.d_min + self.cfg.rho_b + self.cfg.confidence_inflation
        base_axes = self.cfg.body_axes if self.cfg.envelope != "sphere" else (self.cfg.robot_radius, self.cfg.robot_radius)
        if ghost_count <= 0:
            return
        for marker_idx, i in enumerate(range(1, len(states), stride)):
            if marker_idx >= ghost_count:
                break
            pred_state = states[i]
            alpha = max(0.16, (0.62 if self.visual_debug else 0.38) - 0.10 * marker_idx)
            body = body_support_points(
                pred_state,
                base_axes,
                64,
                self.cfg.envelope,
                self.cfg.robot_radius,
                self.cfg.body_shape_exponent,
            )
            body = np.vstack((body, body[0]))
            arr.markers.append(line_marker("tvdhocbf_prediction_body", 520 + marker_idx, body, marker_color(0.00, 0.62, 0.88, alpha), 0.030, z + 0.02 * marker_idx))
            if self.visual_debug and is_rsm_mode(self.cfg.constraint_mode):
                reserve = self.dynamic_reserve(pred_state)
                axes = (base_axes[0] + safety_pad + reserve, base_axes[1] + safety_pad + reserve)
                radius = self.cfg.robot_radius + safety_pad + reserve
                recover = body_support_points(pred_state, axes, 72, self.cfg.envelope, radius, self.cfg.body_shape_exponent)
                recover = np.vstack((recover, recover[0]))
                arr.markers.append(line_marker("tvdhocbf_prediction_recovery", 560 + marker_idx, recover, marker_color(1.00, 0.55, 0.02, max(0.14, alpha - 0.23)), 0.030, z + 0.05 + 0.02 * marker_idx))

    def add_certificate_bars(self, arr: MarkerArray, state: np.ndarray):
        if not self.last_cert:
            return
        entries = [
            ("c", float(self.last_cert.get("c_body", float("nan"))), marker_color(0.00, 0.72, 0.90, 0.92)),
            ("D", float(self.last_cert.get("delta", float("nan"))), marker_color(1.00, 0.58, 0.02, 0.92)),
            ("h", float(self.last_cert.get("h_R", float("nan"))), self.safety_color(float(self.last_cert.get("h_R", float("nan"))), 0.96)),
            ("r", float(self.last_cert.get("dynamic_reserve", float("nan"))), marker_color(1.00, 0.42, 0.02, 0.92)),
        ]
        base_x = state[0] + 0.95
        base_y = state[1] - 1.20
        z = self.goal[2] + 1.35
        max_len = 1.25
        for i, (_name, value, color) in enumerate(entries):
            row_y = base_y + 0.20 * i
            arr.markers.append(cube_marker("tvdhocbf_certificate_baseline", 600 + i, (base_x + max_len * 0.5, row_y, z), (max_len, 0.055, 0.055), marker_color(0.12, 0.12, 0.12, 0.20)))
            if not np.isfinite(value):
                continue
            gain = self.reserve_visual_gain if _name == "r" else 3.2
            length = float(np.clip(abs(value) * gain, 0.045, max_len))
            if value >= 0.0:
                x = base_x + 0.5 * length
            else:
                x = base_x - 0.5 * length
                color = marker_color(0.92, 0.06, 0.05, 0.96)
            arr.markers.append(cube_marker("tvdhocbf_certificate_bar", 620 + i, (x, row_y, z + 0.04), (length, 0.095, 0.095), color))

    def visual_timer(self, _event):
        if rospy.is_shutdown():
            return
        if self.odom is None:
            try:
                self.publish_goal_marker()
            except rospy.ROSException:
                pass
            return
        state = self.state_from_odom()
        try:
            if self.history:
                hist = np.vstack(self.history)
                self.path_pub.publish(path_msg(hist[:, :2], self.goal[2]))
            self.publish_goal_marker()
            self.publish_planner_markers(state)
            self.publish_metrics(state)
        except rospy.ROSException:
            pass

    def publish_goal_marker(self):
        arr = MarkerArray()
        if self.show_ground_plane:
            arr.markers.append(self.ground_plane_marker())
        m = Marker()
        m.header = header()
        m.ns = "goal"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position = pt(self.goal[0], self.goal[1], self.goal[2])
        m.pose.orientation.w = 1.0
        m.scale.x = 0.45
        m.scale.y = 0.45
        m.scale.z = 0.45
        m.color = marker_color(0.95, 0.15, 0.10, 1.0)
        arr.markers.append(m)
        self.goal_viz_pub.publish(arr)

    def ground_plane_marker(self) -> Marker:
        msg = Marker()
        msg.header = header()
        msg.ns = "tvdhocbf_ground_plane"
        msg.id = 0
        msg.type = Marker.CUBE
        msg.action = Marker.ADD
        msg.pose.position = pt(self.ground_center_x, self.ground_center_y, -0.045)
        msg.pose.orientation.w = 1.0
        msg.scale.x = max(0.1, self.ground_size_x)
        msg.scale.y = max(0.1, self.ground_size_y)
        msg.scale.z = 0.035
        msg.color = marker_color(0.50, 0.52, 0.54, float(np.clip(self.ground_alpha, 0.0, 1.0)))
        msg.lifetime = rospy.Duration(0.0)
        return msg

    def publish_planner_markers(self, state: np.ndarray):
        arr = MarkerArray()
        if self.show_uav_model:
            arr.markers.append(self.uav_mesh_marker())
        if self.show_perception_markers:
            self.add_perception_markers(arr, state)
        if len(self.history) >= 2:
            hist = np.vstack(self.history)
            arr.markers.append(
                line_marker(
                    "tvdhocbf_executed_traj",
                    31,
                    hist[:, :2],
                    marker_color(0.00, 0.86, 0.50, self.executed_line_alpha),
                    self.executed_line_width,
                    self.goal[2] + 0.16,
                )
            )
        if self.show_prediction_tube:
            self.add_prediction_markers(arr)
        if self.global_path is not None and len(self.global_path) >= 2:
            arr.markers.append(
                line_marker(
                    "tvdhocbf_guide",
                    0,
                    self.global_path,
                    marker_color(0.36, 0.38, 0.40, self.guide_line_alpha),
                    self.guide_line_width,
                    self.goal[2],
                )
            )
        if self.show_safety_envelope:
            self.add_envelope_markers(arr, state)
        if self.show_nearest_clearance:
            self.add_nearest_clearance_marker(arr, state)
        if self.show_certificate_bars:
            self.add_certificate_bars(arr, state)
        centers = np.vstack([o.center for o in self.last_obstacles]) if self.last_obstacles else np.zeros((0, 2))
        if self.show_perception_markers and len(centers):
            arr.markers.append(sphere_list("tvdhocbf_local_obstacles", 3, centers, marker_color(1.0, 0.65, 0.05, 0.75), 0.22, self.goal[2]))
        self.viz_pub.publish(arr)
        self.points_pub.publish(arr)

    def uav_mesh_marker(self) -> Marker:
        msg = Marker()
        msg.header = header()
        msg.ns = "tvdhocbf_uav_model"
        msg.id = 0
        msg.type = Marker.MESH_RESOURCE
        msg.action = Marker.ADD
        msg.mesh_resource = self.uav_mesh_resource
        msg.mesh_use_embedded_materials = True
        if self.odom is not None:
            msg.pose = self.odom.pose.pose
        else:
            msg.pose.position = pt(0.0, 0.0, self.flight_z)
            msg.pose.orientation.w = 1.0
        msg.scale.x = self.uav_mesh_scale
        msg.scale.y = self.uav_mesh_scale
        msg.scale.z = self.uav_mesh_scale
        msg.color = marker_color(1.0, 1.0, 1.0, 1.0)
        return msg

    def add_envelope_markers(self, arr: MarkerArray, state: np.ndarray):
        z = self.goal[2] + 0.05
        if self.cfg.envelope == "point":
            arr.markers.append(sphere_list("tvdhocbf_point_body", 21, state[:2].reshape(1, 2), marker_color(0.35, 0.35, 0.35, 0.9), 0.18, z))
            return

        if self.cfg.envelope == "sphere":
            axes = (self.cfg.robot_radius, self.cfg.robot_radius)
            color = marker_color(0.45, 0.45, 0.45, 0.18)
            ring_color = marker_color(0.35, 0.35, 0.35, 0.95)
            label = "fixed spherical envelope"
        else:
            axes = self.cfg.body_axes
            color = marker_color(0.00, 0.78, 0.35, 0.22)
            ring_color = marker_color(0.00, 0.70, 0.35, 1.0)
            label = "RSM superellipsoid envelope" if self.cfg.envelope == "superellipse" else "RSM body envelope"

        arr.markers.append(ellipsoid_marker("tvdhocbf_body_envelope", 20, state[:2], state[4], axes, z, color, 0.16))
        body = body_support_points(state, axes, 80, self.cfg.envelope, self.cfg.robot_radius, self.cfg.body_shape_exponent)
        body = np.vstack((body, body[0]))
        arr.markers.append(line_marker("tvdhocbf_body_boundary", 1, body, ring_color, 0.08, z + 0.04))

        safety_pad = self.cfg.d_min + self.cfg.rho_b + self.cfg.confidence_inflation
        margin_axes = (axes[0] + safety_pad, axes[1] + safety_pad)
        margin_radius = self.cfg.robot_radius + safety_pad
        margin = body_support_points(state, margin_axes, 96, self.cfg.envelope, margin_radius, self.cfg.body_shape_exponent)
        margin = np.vstack((margin, margin[0]))
        arr.markers.append(line_marker("tvdhocbf_safety_margin", 22, margin, marker_color(0.0, 0.72, 0.90, 0.85), 0.045, z + 0.10))

        speed = float(np.linalg.norm(state[2:4]))
        if speed > 0.04:
            vel_end = state[:2] + 0.34 * state[2:4]
            arr.markers.append(line_marker("tvdhocbf_velocity_vector", 24, np.vstack((state[:2], vel_end)), marker_color(0.04, 0.34, 1.0, 0.90), 0.075, z + 0.22))

        if is_rsm_mode(self.cfg.constraint_mode):
            reserve = self.dynamic_reserve(state)
            visual_reserve = float(np.clip(reserve * self.reserve_visual_gain, 0.0, self.reserve_visual_max))
            recover_axes = (margin_axes[0] + visual_reserve, margin_axes[1] + visual_reserve)
            recover = body_support_points(state, recover_axes, 128, self.cfg.envelope, margin_radius + visual_reserve, self.cfg.body_shape_exponent)
            recover = np.vstack((recover, recover[0]))
            reserve_ratio = float(np.clip(visual_reserve / max(self.reserve_visual_max, 1e-6), 0.0, 1.0))
            tension = self.rsm_tension()
            reserve_width = 0.055 + 0.060 * reserve_ratio + 0.070 * tension
            arr.markers.append(line_marker("tvdhocbf_recovery_margin", 23, recover, self.reserve_color(reserve_ratio, tension), reserve_width, z + 0.16))
            if speed > 0.04:
                brake_dir = -state[2:4] / max(speed, 1e-6)
                brake_len = float(np.clip(0.45 + 1.4 * visual_reserve, 0.45, 1.85))
                brake_end = state[:2] + brake_len * brake_dir
                arr.markers.append(line_marker("tvdhocbf_recovery_direction", 25, np.vstack((state[:2], brake_end)), self.reserve_color(reserve_ratio, tension), 0.065 + 0.030 * reserve_ratio + 0.050 * tension, z + 0.26))
            label = "RSM envelope + dynamic reserve"

        if self.cfg.envelope in {"ellipse", "superellipse"}:
            support = body_support_points(state, self.cfg.body_axes, 16, self.cfg.envelope, self.cfg.robot_radius, self.cfg.body_shape_exponent)
            point_alpha = 0.95 if self.visual_debug else 0.62
            point_scale = 0.12 if self.visual_debug else 0.075
            arr.markers.append(sphere_list("tvdhocbf_support_points", 2, support, marker_color(1.0, 0.20, 0.08, point_alpha), point_scale, z + 0.13))
            if self.visual_debug:
                for i, s in enumerate(support):
                    arr.markers.append(line_marker("tvdhocbf_support_rays", 40 + i, np.vstack((state[:2], s)), marker_color(1.0, 0.20, 0.08, 0.38), 0.022, z + 0.12))

        arr.markers.append(text_marker("tvdhocbf_envelope_label", 0, label, (state[0] + 0.7, state[1] - 0.7, z + 0.55)))

    def publish_metrics(self, state: np.ndarray):
        arr = MarkerArray()
        fr = 100.0 * np.mean(self.feasible_hist) if self.feasible_hist else 100.0
        solve_ms = np.mean(self.solve_ms_hist[-10:]) if self.solve_ms_hist else 0.0
        clearance = self.clearance_hist[-1] if self.clearance_hist else float("nan")
        d_lower = self.d_lower_hist[-1] if self.d_lower_hist else float("nan")
        barrier = self.barrier_hist[-1] if self.barrier_hist else float("nan")
        rsm = self.recoverable_hist[-1] if self.recoverable_hist else float("nan")
        gamma = self.gamma_hist[-1] if self.gamma_hist else float("nan")
        reserve = self.reserve_hist[-1] if self.reserve_hist else 0.0
        delta = self.delta_hist[-1] if self.delta_hist else float("nan")
        slack = self.slack_hist[-1] if self.slack_hist else 0.0
        title = self.cfg.label
        finite_margins = [v for v in (barrier, delta, rsm) if np.isfinite(v)]
        min_cert = min(finite_margins) if finite_margins else float("nan")
        if np.isfinite(min_cert) and min_cert < 0.0:
            cert_state = "VIOL"
        elif np.isfinite(min_cert) and min_cert < 0.05:
            cert_state = "LOW"
        else:
            cert_state = "OK"
        txt = (
            "%s on SUPER/MARSIM\n"
            "cert: %s   goal: %.1f %.1f %.1f   local pts: %d   guide pts: %d\n"
            "d_lower: %.2f   c_body: %.2f   Gamma: %.2f   V_r: %.3f   Delta: %.2f   hR: %.2f\n"
            "raw clearance: %.2f m   reserve: %.2f   eps: %.3f   FR: %.1f%%   solve: %.1f ms"
            % (
                title,
                cert_state,
                self.goal[0],
                self.goal[1],
                self.goal[2],
                len(self.local_cloud),
                len(self.global_path) if self.global_path is not None else 0,
                d_lower,
                barrier,
                gamma,
                self.energy_hist[-1] if self.energy_hist else float("nan"),
                delta,
                rsm,
                clearance,
                reserve,
                slack,
                fr,
                solve_ms,
            )
        )
        arr.markers.append(text_marker("tvdhocbf_metrics", 0, txt, (state[0] + 1.0, state[1] + 1.0, self.goal[2] + 2.6)))
        self.metrics_pub.publish(arr)


def main():
    rospy.init_node("tv_dhocbf_super")
    TVDHOCBFSuperNode()
    rospy.spin()


if __name__ == "__main__":
    main()
