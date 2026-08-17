#!/usr/bin/env python3
"""Generate uniformly random tilted-rod PCD maps for SUPER/MARSIM.

The first simulation-validation scene keeps the physics deliberately clean:
every obstacle in the map is a randomly tilted rod.  D1--D4 only change the
number and radius of rods.  There is no pre-cleared middle corridor; apart from
start/goal buffers, rods are scattered over the full field.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


OUT_PREFIX = "tvdhocbf_rsm_clutter_corridor"
SEED = 20260604
X_RANGE = (-7.5, 7.5)
Y_RANGE = (-55.0, 55.0)
START_XY = np.array([0.0, -50.0])
GOAL_XY = np.array([0.0, 45.0])

DENSITIES: Dict[str, Dict[str, float]] = {
    "d1": {
        "traversability": 6.5,
        "rod_count": 190,
        "rod_radius": 0.050,
    },
    "d2": {
        "traversability": 5.4,
        "rod_count": 300,
        "rod_radius": 0.054,
    },
    "d3": {
        "traversability": 4.4,
        "rod_count": 430,
        "rod_radius": 0.058,
    },
    "d4": {
        "traversability": 3.2,
        "rod_count": 580,
        "rod_radius": 0.062,
    },
}

def super_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pcd_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd"


def config_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def keep_start_goal_free(x: float, y: float, radius: float = 3.4) -> bool:
    xy = np.array([x, y], dtype=float)
    return np.linalg.norm(xy - START_XY) > radius and np.linalg.norm(xy - GOAL_XY) > radius


def sample_cylinder_between(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    step: float = 0.085,
    n_theta: int = 10,
) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return np.zeros((0, 3), dtype=np.float32)
    w = axis / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, w))) > 0.94:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(w, helper)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(w, u)
    ts = np.linspace(0.0, 1.0, max(2, int(math.ceil(length / step)) + 1))
    angles = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    pts = []
    for t in ts:
        c = (1.0 - t) * p0 + t * p1
        for a in angles:
            pts.append(c + radius * (math.cos(a) * u + math.sin(a) * v))
    return np.asarray(pts, dtype=np.float32)


def generate_rods(rng: np.random.Generator, cfg: Dict[str, float]) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    chunks: List[np.ndarray] = []
    rods: List[Dict[str, float]] = []
    attempts = 0
    while len(rods) < int(cfg["rod_count"]) and attempts < 10 * int(cfg["rod_count"]):
        attempts += 1
        x = float(rng.uniform(X_RANGE[0] + 0.35, X_RANGE[1] - 0.35))
        y = float(rng.uniform(Y_RANGE[0] + 2.0, Y_RANGE[1] - 2.0))
        if not keep_start_goal_free(x, y):
            continue
        height = float(rng.uniform(2.05, 3.90))
        lean_len = float(rng.uniform(0.45, 2.10))
        yaw = float(rng.uniform(-math.pi, math.pi))
        z0 = float(rng.uniform(0.03, 0.18))
        p0 = np.array([x, y, z0], dtype=float)
        p1 = np.array([x + lean_len * math.cos(yaw), y + lean_len * math.sin(yaw), height], dtype=float)
        p1[0] = float(np.clip(p1[0], X_RANGE[0] - 0.8, X_RANGE[1] + 0.8))
        p1[1] = float(np.clip(p1[1], Y_RANGE[0] + 0.4, Y_RANGE[1] - 0.4))
        if not keep_start_goal_free(float(p1[0]), float(p1[1]), radius=2.8):
            continue
        radius = float(cfg["rod_radius"] * rng.uniform(0.82, 1.22))
        chunks.append(sample_cylinder_between(p0, p1, radius, step=0.095, n_theta=8))
        rods.append(
            {
                "x": x,
                "y": y,
                "z0": z0,
                "x1": float(p1[0]),
                "y1": float(p1[1]),
                "z1": float(p1[2]),
                "radius": radius,
                "shade": float(rng.uniform(0.0, 1.0)),
            }
        )
    return chunks, rods


def generate_scene(density: str, cfg: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, object]]:
    rng = np.random.default_rng(SEED + sum(ord(c) for c in density))
    chunks: List[np.ndarray] = []
    rod_chunks, rods = generate_rods(rng, cfg)
    chunks.extend(rod_chunks)

    points = np.vstack(chunks).astype(np.float32)
    points = points[np.isfinite(points).all(axis=1)]
    points = points[
        (points[:, 0] >= X_RANGE[0] - 1.0)
        & (points[:, 0] <= X_RANGE[1] + 1.0)
        & (points[:, 1] >= Y_RANGE[0])
        & (points[:, 1] <= Y_RANGE[1])
        & (points[:, 2] >= -0.05)
        & (points[:, 2] <= 4.0)
    ]
    points = np.unique(np.round(points, 4), axis=0)
    rng.shuffle(points)

    guide_y = np.linspace(-50.0, 45.0, 80)
    guide = [[0.0, float(y), 1.52] for y in guide_y]
    meta = {
        "name": OUT_PREFIX,
        "density": density,
        "traversability": cfg["traversability"],
        "placement": "uniform_random",
        "start": [0.0, -50.0, 1.5],
        "goal": [0.0, 45.0, 1.5],
        "guide": guide,
        "rods": rods,
    }
    return points, meta


def write_ascii_xyz_pcd(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "\n".join(
        [
            "# .PCD v0.7 - Point Cloud Data file format",
            "VERSION 0.7",
            "FIELDS x y z",
            "SIZE 4 4 4",
            "TYPE F F F",
            "COUNT 1 1 1",
            "WIDTH %d" % len(points),
            "HEIGHT 1",
            "VIEWPOINT 0 0 0 1 0 0 0",
            "POINTS %d" % len(points),
            "DATA ascii",
        ]
    )
    with path.open("w", encoding="utf-8") as f:
        f.write(header + "\n")
        np.savetxt(f, points, fmt="%.4f %.4f %.4f")


def write_config(path: Path, pcd_name: str) -> None:
    text = """#############| For UAV simulation |#################################
pcd_name: "%s"
mesh_resource: "package://perfect_drone_sim/meshes/yunque-M.dae"
init_position:
  x: 0
  y: -50
  z: 1.5
init_yaw: 1.5708

#############|For LiDAR perception simulation |#################################
is_360lidar: true
polar_resolution: 0.4
downsample_res: 0.1
vertical_fov: 178.0
sensing_blind: 0.1
sensing_horizon: 20
sensing_rate: 10
print_time_consumption: false
lidar_type: 2
depth_image_en: false
""" % pcd_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for density, cfg in DENSITIES.items():
        points, meta = generate_scene(density, cfg)
        pcd_name = "%s_%s.pcd" % (OUT_PREFIX, density)
        yaml_name = "%s_%s.yaml" % (OUT_PREFIX, density)
        json_name = "%s_%s.json" % (OUT_PREFIX, density)
        write_ascii_xyz_pcd(pcd_dir() / pcd_name, points)
        write_config(config_dir() / yaml_name, pcd_name)
        (config_dir() / json_name).write_text(json.dumps(meta, indent=2), encoding="utf-8")

        d_start = float(np.linalg.norm(points[:, :2] - START_XY.reshape(1, 2), axis=1).min())
        d_goal = float(np.linalg.norm(points[:, :2] - GOAL_XY.reshape(1, 2), axis=1).min())
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        print(
            "%s: points=%d rods=%d trav=%.1f bounds=[%.1f %.1f %.1f]..[%.1f %.1f %.1f] start_clear=%.2f goal_clear=%.2f"
            % (
                density,
                len(points),
                len(meta["rods"]),
                cfg["traversability"],
                lo[0],
                lo[1],
                lo[2],
                hi[0],
                hi[1],
                hi[2],
                d_start,
                d_goal,
            )
        )


if __name__ == "__main__":
    main()
