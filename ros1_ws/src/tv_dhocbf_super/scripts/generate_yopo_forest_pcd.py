#!/usr/bin/env python3
"""Generate a YOPO-style random forest for SUPER/MARSIM.

The map follows the spirit of YOPO's maze_type=5 random forest: trees are
placed on a jittered Poisson-like grid, with random scale and slight lean.  The
PCD is used by MARSIM for sensing, while a sidecar JSON file stores tree
metadata for prettier RViz marker rendering.
"""

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


OUT_PREFIX = "tvdhocbf_yopo_forest"
SEED = 20260526
X_RANGE = (-7.4, 7.4)
Y_RANGE = (-54.0, 54.0)
START_XY = np.array([0.0, -50.0])
GOAL_XY = np.array([0.0, 45.0])

DENSITY_LEVELS: Dict[str, Dict[str, float]] = {
    "d1": {"tree_dist": 5.20, "keep": 0.92, "radius_scale": 0.92},
    "d2": {"tree_dist": 4.50, "keep": 0.94, "radius_scale": 0.98},
    "d3": {"tree_dist": 3.90, "keep": 0.96, "radius_scale": 1.04},
    "d4": {"tree_dist": 3.35, "keep": 0.98, "radius_scale": 1.10},
}


def super_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pcd_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd"


def config_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def sample_cylinder(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    step: float = 0.070,
    n_theta: int = 14,
) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return np.zeros((0, 3), dtype=np.float32)
    w = axis / length
    helper = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(helper, w))) > 0.92:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(w, helper)
    u /= max(np.linalg.norm(u), 1e-9)
    v = np.cross(w, u)
    ts = np.linspace(0.0, 1.0, max(2, int(np.ceil(length / step)) + 1))
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    pts = []
    for t in ts:
        c = (1.0 - t) * p0 + t * p1
        for a in angles:
            pts.append(c + radius * (np.cos(a) * u + np.sin(a) * v))
    return np.asarray(pts, dtype=np.float32)


def sample_ellipsoid_shell(
    center: Sequence[float],
    radii: Sequence[float],
    n_theta: int = 26,
    n_phi: int = 12,
) -> np.ndarray:
    cx, cy, cz = center
    rx, ry, rz = radii
    pts = []
    # Avoid the poles being too sparse by using staggered latitude rings.
    for i, phi in enumerate(np.linspace(0.18, np.pi - 0.18, n_phi)):
        offset = (i % 2) * np.pi / n_theta
        for theta in np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False) + offset:
            pts.append(
                [
                    cx + rx * np.sin(phi) * np.cos(theta),
                    cy + ry * np.sin(phi) * np.sin(theta),
                    cz + rz * np.cos(phi),
                ]
            )
    return np.asarray(pts, dtype=np.float32)


def generate_tree_specs(density_name: str, density: Dict[str, float]) -> List[Dict[str, float]]:
    rng = np.random.default_rng(SEED + sum(ord(c) for c in density_name))
    dist = density["tree_dist"]
    specs: List[Dict[str, float]] = []
    xs = np.arange(X_RANGE[0], X_RANGE[1] + dist, dist)
    ys = np.arange(Y_RANGE[0], Y_RANGE[1] + dist, dist)
    for ix, x0 in enumerate(xs):
        for iy, y0 in enumerate(ys):
            if rng.random() > density["keep"]:
                continue
            x = x0 + rng.uniform(0.18, 0.82) * dist
            y = y0 + rng.uniform(0.18, 0.82) * dist
            if not (X_RANGE[0] <= x <= X_RANGE[1] and Y_RANGE[0] <= y <= Y_RANGE[1]):
                continue
            xy = np.array([x, y])
            if np.linalg.norm(xy - START_XY) < 2.8 or np.linalg.norm(xy - GOAL_XY) < 2.8:
                continue
            scale = rng.uniform(0.78, 1.18)
            radius = density["radius_scale"] * scale * rng.uniform(0.13, 0.22)
            height = scale * rng.uniform(3.7, 5.2)
            lean_len = rng.uniform(0.0, 0.35) * scale
            lean_yaw = rng.uniform(-np.pi, np.pi)
            canopy_rx = scale * rng.uniform(0.74, 1.10)
            canopy_ry = scale * rng.uniform(0.70, 1.04)
            canopy_rz = scale * rng.uniform(0.45, 0.68)
            canopy_z = height * rng.uniform(0.82, 0.91) + canopy_rz * 0.35
            green = rng.uniform(0.0, 1.0)
            specs.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "radius": float(radius),
                    "height": float(height),
                    "lean_x": float(lean_len * np.cos(lean_yaw)),
                    "lean_y": float(lean_len * np.sin(lean_yaw)),
                    "canopy_rx": float(canopy_rx),
                    "canopy_ry": float(canopy_ry),
                    "canopy_rz": float(canopy_rz),
                    "canopy_z": float(canopy_z),
                    "green_mix": float(green),
                }
            )
    return specs


def make_scene(density_name: str, density: Dict[str, float]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    specs = generate_tree_specs(density_name, density)
    chunks: List[np.ndarray] = []
    for spec in specs:
        x, y = spec["x"], spec["y"]
        p0 = (x, y, 0.0)
        p1 = (x + spec["lean_x"], y + spec["lean_y"], spec["height"])
        chunks.append(sample_cylinder(p0, p1, spec["radius"]))
        chunks.append(
            sample_ellipsoid_shell(
                (x + 0.72 * spec["lean_x"], y + 0.72 * spec["lean_y"], spec["canopy_z"]),
                (spec["canopy_rx"], spec["canopy_ry"], spec["canopy_rz"]),
            )
        )
        # A few high branches give the cloud a tree-like silhouette but stay
        # mostly above the 1.5 m flight corridor used for safety constraints.
        for branch_i in range(3):
            yaw = 2.0 * np.pi * (branch_i / 3.0 + 0.11 * spec["green_mix"])
            z = spec["height"] * (0.58 + 0.10 * branch_i)
            length = 0.55 * max(spec["canopy_rx"], spec["canopy_ry"])
            start = (x + 0.25 * spec["lean_x"], y + 0.25 * spec["lean_y"], z)
            end = (start[0] + length * np.cos(yaw), start[1] + length * np.sin(yaw), z + 0.25)
            chunks.append(sample_cylinder(start, end, max(0.025, 0.22 * spec["radius"]), step=0.09, n_theta=8))
    points = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, 3), dtype=np.float32)
    return points, specs


def write_ascii_xyz_pcd(path: Path, points: np.ndarray) -> None:
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
        np.savetxt(f, points, fmt="%.5f %.5f %.5f")


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
polar_resolution: 0.35
downsample_res: 0.1
vertical_fov: 178.0
sensing_blind: 0.1
sensing_horizon: 20
sensing_rate: 10
print_time_consumption: false
lidar_type: 2
depth_image_en: false
""" % pcd_name
    path.write_text(text, encoding="utf-8")


def main() -> None:
    pcd_dir().mkdir(parents=True, exist_ok=True)
    config_dir().mkdir(parents=True, exist_ok=True)
    for density_name, density in DENSITY_LEVELS.items():
        points, specs = make_scene(density_name, density)
        pcd_name = "%s_%s.pcd" % (OUT_PREFIX, density_name)
        yaml_name = "%s_%s.yaml" % (OUT_PREFIX, density_name)
        json_name = "%s_%s.json" % (OUT_PREFIX, density_name)
        write_ascii_xyz_pcd(pcd_dir() / pcd_name, points)
        write_config(config_dir() / yaml_name, pcd_name)
        (config_dir() / json_name).write_text(json.dumps({"trees": specs}, indent=2), encoding="utf-8")
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        print(
            "%s: trees=%d points=%d bounds=[%.1f %.1f %.1f]..[%.1f %.1f %.1f]"
            % (density_name, len(specs), len(points), lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        )


if __name__ == "__main__":
    main()
