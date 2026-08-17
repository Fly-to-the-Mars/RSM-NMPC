#!/usr/bin/env python3
"""Generate the deterministic TV-DHOCBF rescue arena PCD for SUPER/MARSIM."""

import math
from pathlib import Path

import numpy as np


SEED = 20260525
SPACING = 0.075
PCD_NAME = "tvdhocbf_rescue_arena.pcd"


def rot2(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=float)


def grid_axis(length: float, spacing: float) -> np.ndarray:
    count = max(2, int(math.ceil(length / spacing)) + 1)
    return np.linspace(-0.5 * length, 0.5 * length, count)


def add_box(points, center, size, yaw=0.0, spacing=SPACING):
    cx, cy, cz = center
    sx, sy, sz = size
    xs = grid_axis(sx, spacing)
    ys = grid_axis(sy, spacing)
    zs = grid_axis(sz, spacing)
    r = rot2(yaw)

    def push(local):
        xy = r @ np.array([local[0], local[1]], dtype=float)
        points.append((cx + xy[0], cy + xy[1], cz + local[2]))

    for x in (-0.5 * sx, 0.5 * sx):
        for y in ys:
            for z in zs:
                push((x, y, z))
    for y in (-0.5 * sy, 0.5 * sy):
        for x in xs:
            for z in zs:
                push((x, y, z))
    for z in (-0.5 * sz, 0.5 * sz):
        for x in xs:
            for y in ys:
                push((x, y, z))


def add_cylinder(points, center_xy, radius, height, spacing=SPACING, z0=0.0):
    cx, cy = center_xy
    n_theta = max(18, int(math.ceil(2.0 * math.pi * radius / spacing)))
    n_z = max(2, int(math.ceil(height / spacing)) + 1)
    theta = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    zvals = np.linspace(z0, z0 + height, n_z)
    for t in theta:
        x = cx + radius * math.cos(t)
        y = cy + radius * math.sin(t)
        for z in zvals:
            points.append((x, y, z))
    # Light cap rings help MARSIM render the columns as solid objects.
    for rr in np.linspace(0.0, radius, max(3, int(radius / spacing))):
        for t in theta:
            points.append((cx + rr * math.cos(t), cy + rr * math.sin(t), z0 + height))


def add_gate(points, origin, yaw, total_width=7.2, gap=1.08, thickness=0.28, height=2.75):
    ox, oy = origin
    seg = 0.5 * (total_width - gap)
    r = rot2(yaw)
    for sign in (-1.0, 1.0):
        local_center = np.array([sign * (0.5 * gap + 0.5 * seg), 0.0])
        world = np.array([ox, oy]) + r @ local_center
        add_box(points, (world[0], world[1], 0.5 * height), (seg, thickness, height), yaw=yaw)
    for sign in (-1.0, 1.0):
        local_center = np.array([sign * 0.5 * gap, 0.0])
        world = np.array([ox, oy]) + r @ local_center
        add_cylinder(points, world, 0.12, height, spacing=0.06)


def build_scene() -> np.ndarray:
    points = []

    # Long side boundaries make the arena visually legible without cluttering the route.
    add_box(points, (-8.6, 17.0, 1.2), (0.26, 54.0, 2.4), spacing=0.09)
    add_box(points, (8.6, 17.0, 1.2), (0.26, 54.0, 2.4), spacing=0.09)
    add_box(points, (0.0, -10.0, 1.2), (17.2, 0.26, 2.4), spacing=0.09)
    add_box(points, (0.0, 44.0, 1.2), (17.2, 0.26, 2.4), spacing=0.09)

    # S-bend pillar field: sparse but route-shaping.
    s_gates = [
        (-3.75, -0.6, 1.15, 8.5, 0.25, 2.3, 0.0),
        (3.75, 4.4, 1.15, 8.5, 0.25, 2.3, 0.0),
        (-3.55, 9.1, 1.15, 7.8, 0.25, 2.3, 0.0),
    ]
    for x, y, z, sx, sy, sz, yaw in s_gates:
        add_box(points, (x, y, z), (sx, sy, sz), yaw=yaw, spacing=0.085)

    pillars = [
        (-2.6, -1.8, 0.40),
        (2.3, 0.8, 0.36),
        (-2.0, 3.4, 0.38),
        (2.6, 6.1, 0.40),
        (-2.4, 8.6, 0.36),
        (1.7, 10.9, 0.34),
        (-5.0, 2.0, 0.42),
        (5.1, 7.8, 0.42),
    ]
    for x, y, r in pillars:
        add_cylinder(points, (x, y), r, 3.0)

    # Debris piles along the S-bend margins add irregular geometry, but not random density.
    debris = [
        (-4.4, -3.0, 0.42, 0.8, 0.7, 0.84, 0.10),
        (4.5, 3.5, 0.45, 0.9, 0.75, 0.9, -0.24),
        (-4.1, 7.0, 0.36, 1.1, 0.55, 0.72, 0.32),
    ]
    for x, y, z, sx, sy, sz, yaw in debris:
        add_box(points, (x, y, z), (sx, sy, sz), yaw=yaw, spacing=0.065)

    # Tilted narrow gate: the key embodiment-aware passage.
    add_gate(points, (0.0, 14.9), math.radians(24.0), total_width=15.8, gap=1.75)

    # Occluded L-corridor: short walls and a central occluder reveal the safe set locally.
    add_box(points, (-2.15, 23.4, 1.35), (0.30, 9.5, 2.7), yaw=0.0, spacing=0.08)
    add_box(points, (2.15, 27.4, 1.35), (0.30, 9.2, 2.7), yaw=0.0, spacing=0.08)
    add_box(points, (0.65, 23.4, 1.1), (1.0, 1.65, 2.2), yaw=math.radians(8.0), spacing=0.065)
    add_box(points, (-0.95, 29.8, 0.9), (1.2, 0.75, 1.8), yaw=math.radians(-18.0), spacing=0.065)

    # Inspection slot: low slanted side walls encourage full-body clearance reasoning.
    add_box(points, (-1.08, 34.1, 0.95), (0.20, 4.4, 1.9), yaw=math.radians(8.0), spacing=0.07)
    add_box(points, (1.10, 34.7, 0.95), (0.20, 4.4, 1.9), yaw=math.radians(-8.0), spacing=0.07)
    add_box(points, (3.25, 34.0, 0.65), (1.0, 0.55, 1.3), yaw=math.radians(24.0), spacing=0.065)

    # Exit sprint visual gates, safely off the central racing line.
    for x in (-3.4, 3.4):
        add_cylinder(points, (x, 39.5), 0.16, 2.4, spacing=0.06)

    arr = np.array(points, dtype=np.float32)
    arr = arr[np.isfinite(arr).all(axis=1)]
    arr = np.unique(np.round(arr, 4), axis=0)
    rng = np.random.default_rng(SEED)
    rng.shuffle(arr)
    return arr


def write_pcd(path: Path, points: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {len(points)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(points)}\n"
        "DATA ascii\n"
    )
    with path.open("w") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.4f %.4f %.4f")


def main():
    script_path = Path(__file__).resolve()
    src_dir = script_path.parents[2]
    pcd_path = src_dir / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd" / PCD_NAME
    pts = build_scene()
    write_pcd(pcd_path, pts)

    start = np.array([0.0, -8.0])
    goal = np.array([0.0, 42.0])
    d_start = np.linalg.norm(pts[:, :2] - start.reshape(1, 2), axis=1).min()
    d_goal = np.linalg.norm(pts[:, :2] - goal.reshape(1, 2), axis=1).min()
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    print(f"Wrote {pcd_path}")
    print(f"points={len(pts)} range_x=[{lo[0]:.2f},{hi[0]:.2f}] range_y=[{lo[1]:.2f},{hi[1]:.2f}] range_z=[{lo[2]:.2f},{hi[2]:.2f}]")
    print(f"nearest_start_xy={d_start:.2f} nearest_goal_xy={d_goal:.2f}")


if __name__ == "__main__":
    main()
