#!/usr/bin/env python3
"""Generate the custom TVD-RSM agile forest benchmark for SUPER/MARSIM.

The scene is intentionally not a clone of SUPER's random forest. It keeps the
same PCD/MARSIM interface, but uses a structured dense layout:

1. takeoff buffer,
2. braided S-bend slalom,
3. biased tilted keyhole gates,
4. occluding comb corridor,
5. terminal speed lattice.

The design is dense enough for perception-driven local planning, yet each
section stresses a different part of the RSM-NMPC safety model.
"""

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


OUT_PREFIX = "tvdhocbf_agile_forest"
SEED = 20260526

DENSITY_LEVELS: Dict[str, Dict[str, float]] = {
    "d1": {"extra": 0.95, "lane": 1.30, "gate": 0.70},
    "d2": {"extra": 1.18, "lane": 1.12, "gate": 0.58},
    "d3": {"extra": 1.45, "lane": 0.98, "gate": 0.48},
    "d4": {"extra": 1.72, "lane": 0.88, "gate": 0.42},
}


def super_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pcd_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd"


def config_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def centerline_x(y: float) -> float:
    return 1.05 * np.sin((y + 31.0) / 8.5) + 0.34 * np.sin((y + 8.0) / 3.6)


def sample_cylinder(
    p0: Sequence[float],
    p1: Sequence[float],
    radius: float,
    step: float = 0.065,
    n_theta: int = 12,
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
    samples = max(2, int(np.ceil(length / step)) + 1)
    ts = np.linspace(0.0, 1.0, samples)
    angles = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    pts = []
    for t in ts:
        c = (1.0 - t) * p0 + t * p1
        for a in angles:
            pts.append(c + radius * (np.cos(a) * u + np.sin(a) * v))
    pts.extend([p0, p1])
    return np.asarray(pts, dtype=np.float32)


def add_rod(
    out: List[np.ndarray],
    xy: Sequence[float],
    height: float,
    radius: float,
    tilt: Sequence[float] = (0.0, 0.0),
    z0: float = 0.0,
) -> None:
    x, y = float(xy[0]), float(xy[1])
    tx, ty = float(tilt[0]), float(tilt[1])
    out.append(sample_cylinder((x, y, z0), (x + tx, y + ty, z0 + height), radius))


def add_wall_rod(
    out: List[np.ndarray],
    center: Sequence[float],
    length: float,
    yaw: float,
    z: float,
    radius: float,
    step: float = 0.09,
) -> None:
    x, y = float(center[0]), float(center[1])
    dx = 0.5 * length * np.cos(yaw)
    dy = 0.5 * length * np.sin(yaw)
    out.append(sample_cylinder((x - dx, y - dy, z), (x + dx, y + dy, z), radius, step=step, n_theta=10))


def add_gate(out: List[np.ndarray], y: float, density: Dict[str, float], rng: np.random.Generator, idx: int) -> None:
    cx = centerline_x(y) + 0.18 * np.sin(0.7 * idx)
    half_gap = density["gate"]
    yaw = 0.35 * (-1.0 if idx % 2 else 1.0)
    post_radius = 0.105 + 0.010 * rng.random()
    for side in (-1.0, 1.0):
        x = cx + side * half_gap
        add_rod(out, (x, y), 3.0, post_radius, tilt=(0.18 * side, 0.25 * np.sin(idx)), z0=0.0)
        add_wall_rod(out, (x + 0.13 * side, y - 0.38), 1.15, yaw + 0.20 * side, 1.45, 0.055)
    add_wall_rod(out, (cx, y + 0.12), 2.0 * half_gap + 0.34, yaw, 2.35, 0.075)
    # A low side rail makes the passage visibly embodied without fully closing it.
    rail_side = -1.0 if idx % 2 else 1.0
    add_wall_rod(out, (cx + rail_side * (half_gap + 0.18), y + 0.42), 1.0, yaw + 0.55, 1.05, 0.052)


def add_slalom(out: List[np.ndarray], density: Dict[str, float], rng: np.random.Generator) -> None:
    lane = density["lane"]
    ys = np.linspace(-42.0, -15.0, int(34 * density["extra"]))
    for i, y in enumerate(ys):
        c = centerline_x(float(y))
        side = -1.0 if i % 2 else 1.0
        for s in (-1.0, 1.0):
            offset = s * (lane + 0.82 + 0.20 * rng.random())
            x = c + offset
            tilt = (0.28 * side * s + rng.normal(scale=0.05), 0.18 * np.cos(0.6 * i))
            add_rod(out, (x, y + rng.normal(scale=0.22)), 2.6 + 0.5 * rng.random(), 0.085 + 0.025 * rng.random(), tilt=tilt)
        if rng.random() < 0.42 * density["extra"]:
            x = c + side * (lane + 0.38 + 0.15 * rng.random())
            add_wall_rod(out, (x, y + 0.35), 0.95 + 0.25 * rng.random(), 0.75 * side, 1.45, 0.045)
        if rng.random() < 0.24 * density["extra"]:
            # Inner leaning twigs sit near the feasible tube and force the
            # controller to use body-aware clearance instead of a wide sphere.
            x = c - side * (lane + 0.05 + 0.08 * rng.random())
            add_wall_rod(out, (x, y - 0.18), 0.70 + 0.18 * rng.random(), -0.95 * side, 1.35, 0.040)


def add_comb_corridor(out: List[np.ndarray], density: Dict[str, float], rng: np.random.Generator) -> None:
    lane = density["lane"]
    ys = np.linspace(7.0, 27.0, int(22 * density["extra"]))
    for i, y in enumerate(ys):
        c = centerline_x(float(y))
        side = -1.0 if i % 2 else 1.0
        root_x = c + side * (lane + 2.0)
        length = 1.55 + 0.28 * density["extra"] + 0.20 * rng.random()
        yaw = np.pi / 2.0 - side * (0.15 + 0.20 * rng.random())
        # Fins protrude from alternating sides, causing local point-cloud updates.
        add_wall_rod(out, (root_x - side * 0.5 * length, y), length, yaw, 1.35, 0.070)
        add_wall_rod(out, (root_x - side * 0.5 * length, y), length, yaw, 2.05, 0.052)
        if rng.random() < 0.55:
            add_rod(out, (root_x, y), 2.4, 0.075, tilt=(-0.20 * side, rng.normal(scale=0.08)))
        if rng.random() < 0.36 * density["extra"]:
            add_wall_rod(out, (c - side * (lane + 0.18), y + 0.28), 0.75, -0.50 * side, 1.55, 0.045)


def add_speed_lattice(out: List[np.ndarray], density: Dict[str, float], rng: np.random.Generator) -> None:
    lane = density["lane"] + 0.22
    count = int(48 * density["extra"])
    for i in range(count):
        y = rng.uniform(29.0, 51.0)
        c = centerline_x(y)
        side = -1.0 if rng.random() < 0.5 else 1.0
        x = c + side * rng.uniform(lane + 0.35, 5.8)
        yaw = rng.uniform(-1.2, 1.2)
        if rng.random() < 0.65:
            add_wall_rod(out, (x, y), rng.uniform(0.75, 1.45), yaw, rng.uniform(0.9, 2.35), rng.uniform(0.040, 0.065))
        else:
            add_rod(out, (x, y), rng.uniform(1.4, 2.9), rng.uniform(0.06, 0.10), tilt=(0.32 * np.cos(yaw), 0.32 * np.sin(yaw)))
        if rng.random() < 0.18 * density["extra"]:
            c = centerline_x(y)
            add_wall_rod(out, (c + side * (lane + 0.18), y + rng.normal(scale=0.35)), rng.uniform(0.55, 0.90), -0.65 * side, 1.42, 0.042)


def add_side_boundaries(out: List[np.ndarray], rng: np.random.Generator) -> None:
    for y in np.linspace(-54.0, 54.0, 92):
        for x in (-7.2, 7.2):
            if rng.random() < 0.86:
                add_rod(out, (x + rng.normal(scale=0.05), y + rng.normal(scale=0.25)), 2.2, 0.055, tilt=(rng.normal(scale=0.10), rng.normal(scale=0.10)))


def add_recovery_pockets(out: List[np.ndarray], density: Dict[str, float], rng: np.random.Generator) -> None:
    """Dense side clusters that leave a narrow recoverable tube between them."""
    lane = density["lane"]
    pocket_centers = [-33.0, -24.0, -16.0, 11.5, 22.5, 36.0]
    for k, y0 in enumerate(pocket_centers):
        c = centerline_x(y0)
        side = -1.0 if k % 2 else 1.0
        for j in range(int(5 * density["extra"])):
            yy = y0 + rng.normal(scale=1.0)
            xx = c + side * rng.uniform(lane + 0.45, lane + 1.35)
            yaw = rng.uniform(-1.3, 1.3)
            add_wall_rod(out, (xx, yy), rng.uniform(0.65, 1.25), yaw, rng.uniform(1.0, 2.1), rng.uniform(0.040, 0.060))
        # Small open pocket on the opposite side supports the recovery story.
        if density["extra"] > 1.0:
            add_rod(out, (c - side * (lane + 1.65), y0), 2.2, 0.070, tilt=(0.15 * side, 0.0))


def make_scene(density_name: str, density: Dict[str, float]) -> np.ndarray:
    rng = np.random.default_rng(SEED + sum(ord(c) for c in density_name))
    chunks: List[np.ndarray] = []

    add_side_boundaries(chunks, rng)
    add_slalom(chunks, density, rng)
    for idx, y in enumerate([-10.0, -3.6, 3.4]):
        add_gate(chunks, y, density, rng, idx)
    add_comb_corridor(chunks, density, rng)
    add_recovery_pockets(chunks, density, rng)
    add_speed_lattice(chunks, density, rng)

    # A few overhead branches are visually useful and do not block the 1.5 m
    # flight corridor; they make the RViz scene less like a flat 2D puzzle.
    for y in np.linspace(-36.0, 42.0, int(10 * density["extra"])):
        c = centerline_x(float(y))
        add_wall_rod(chunks, (c + rng.uniform(-2.0, 2.0), y + rng.normal(scale=0.5)), rng.uniform(1.6, 2.8), rng.uniform(-1.0, 1.0), 2.85, 0.045)

    points = np.vstack(chunks).astype(np.float32)
    # Keep a clear start and goal bubble.
    start = np.array([0.0, -50.0])
    goal = np.array([0.0, 45.0])
    d_start = np.linalg.norm(points[:, :2] - start.reshape(1, 2), axis=1)
    d_goal = np.linalg.norm(points[:, :2] - goal.reshape(1, 2), axis=1)
    points = points[(d_start > 1.25) & (d_goal > 1.25)]
    return points


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
        points = make_scene(density_name, density)
        pcd_name = "%s_%s.pcd" % (OUT_PREFIX, density_name)
        yaml_name = "%s_%s.yaml" % (OUT_PREFIX, density_name)
        write_ascii_xyz_pcd(pcd_dir() / pcd_name, points)
        write_config(config_dir() / yaml_name, pcd_name)
        lo = points.min(axis=0)
        hi = points.max(axis=0)
        print(
            "%s: points=%d bounds=[%.1f %.1f %.1f]..[%.1f %.1f %.1f]"
            % (density_name, len(points), lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        )


if __name__ == "__main__":
    main()
