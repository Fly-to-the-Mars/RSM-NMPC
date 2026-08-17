#!/usr/bin/env python3
"""Generate density variants from SUPER's original dense MARSIM map.

The source map is not redrawn. Instead, this script segments the original
SUPER random-map PCD into ordered obstacle chunks and deterministically keeps a
different fraction of those chunks for each density level. This preserves the
visual language of the SUPER benchmark: long, randomly tilted stick obstacles
in a 15 x 110 m corridor.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


SRC_PCD = "random_map_2_26609.pcd"
OUT_PREFIX = "tvdhocbf_super_density"
SEED = 20260526

# Fractions of segmented SUPER obstacles retained from the original dense map.
DENSITY_FRACTIONS: Dict[str, float] = {
    "d1": 0.25,
    "d2": 0.40,
    "d3": 0.60,
    "d4": 0.80,
}


def intersects_center_flight_lane(segment: np.ndarray) -> bool:
    """Return true if a chunk blocks the nominal low-density center lane.

    D1 is meant to show the easy, low-clutter limit where the proposed method
    can fly almost directly from start to goal.  We therefore keep the random
    SUPER obstacle style but avoid selecting chunks that pierce a very narrow
    lane around x=0 at flight altitude.  Higher density levels are untouched.
    """
    xy = segment[:, :2]
    z = segment[:, 2]
    mask = (
        (np.abs(xy[:, 0]) < 0.52)
        & (xy[:, 1] > -51.0)
        & (xy[:, 1] < 46.0)
        & (np.abs(z - 1.5) < 1.25)
    )
    return bool(np.any(mask))


def super_src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def pcd_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "pcd"


def config_dir() -> Path:
    return super_src_root() / "SUPER" / "mars_uav_sim" / "perfect_drone_sim" / "config"


def read_ascii_xyz_pcd(path: Path) -> np.ndarray:
    lines = path.read_text(errors="ignore").splitlines()
    data_idx = None
    for i, line in enumerate(lines):
        if line.startswith("DATA"):
            if "ascii" not in line:
                raise ValueError("%s is not an ASCII PCD" % path)
            data_idx = i + 1
            break
    if data_idx is None:
        raise ValueError("Missing DATA ascii header in %s" % path)

    points: List[Tuple[float, float, float]] = []
    for line in lines[data_idx:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if np.isfinite([x, y, z]).all():
            points.append((x, y, z))
    if not points:
        raise ValueError("No finite XYZ points in %s" % path)
    return np.asarray(points, dtype=np.float32)


def write_ascii_xyz_pcd(path: Path, points: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32)
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


def obstacle_segments(points: np.ndarray, split_distance: float = 5.0) -> List[np.ndarray]:
    """Split the original SUPER map into obstacle chunks.

    The source PCD is ordered by generated obstacle. Consecutive chunks have a
    much larger XY jump than samples on the same tilted stick, so this simple
    order-aware split preserves individual obstacle geometry better than a 2D
    connected-component pass on the already dense corridor.
    """
    xy_jump = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cuts = np.where(xy_jump > split_distance)[0] + 1
    segments = [seg for seg in np.split(points, cuts) if len(seg) >= 40]
    if len(segments) < 100:
        raise RuntimeError("Unexpectedly few obstacle segments: %d" % len(segments))
    return segments


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
    path.write_text(text, encoding="utf-8")


def main() -> None:
    src = pcd_dir() / SRC_PCD
    points = read_ascii_xyz_pcd(src)
    segments = obstacle_segments(points)
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(segments))

    print("Source:", src)
    print("Finite points:", len(points))
    print("Obstacle chunks:", len(segments))

    for density, fraction in DENSITY_FRACTIONS.items():
        keep_count = max(1, int(round(fraction * len(segments))))
        selected_ids: List[int] = []
        for idx in order:
            if density == "d1" and intersects_center_flight_lane(segments[int(idx)]):
                continue
            selected_ids.append(int(idx))
            if len(selected_ids) >= keep_count:
                break
        if len(selected_ids) < keep_count:
            raise RuntimeError("Not enough obstacle chunks for %s after center-lane filtering" % density)
        keep_ids = set(selected_ids)
        kept = np.vstack([seg for i, seg in enumerate(segments) if i in keep_ids])
        pcd_name = "%s_%s.pcd" % (OUT_PREFIX, density)
        yaml_name = "%s_%s.yaml" % (OUT_PREFIX, density)
        write_ascii_xyz_pcd(pcd_dir() / pcd_name, kept)
        write_config(config_dir() / yaml_name, pcd_name)
        lo = kept.min(axis=0)
        hi = kept.max(axis=0)
        print(
            "%s: chunks=%d points=%d bounds=[%.1f %.1f %.1f]..[%.1f %.1f %.1f]"
            % (density, keep_count, len(kept), lo[0], lo[1], lo[2], hi[0], hi[1], hi[2])
        )


if __name__ == "__main__":
    main()
