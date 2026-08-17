"""Scenario generation and grid guidance for the validation experiments."""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray
    radius: float


@dataclass(frozen=True)
class VisualObject:
    kind: str
    center: np.ndarray
    size: Tuple[float, float, float]
    yaw: float = 0.0
    color: Tuple[float, float, float, float] = (0.45, 0.45, 0.45, 1.0)


@dataclass
class Scenario:
    name: str
    bounds: Tuple[float, float, float, float]
    start: np.ndarray
    goal: np.ndarray
    obstacles: List[Obstacle]
    path: Optional[np.ndarray] = None
    visuals: List[VisualObject] = field(default_factory=list)


def make_dense(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            [1.8, -0.58],
            [3.0, 0.58],
            [4.2, -0.52],
            [5.5, 0.58],
            [6.8, -0.62],
            [8.1, 0.52],
            [9.4, -0.48],
        ],
        dtype=float,
    )
    jitter = rng.normal(scale=[0.10, 0.10], size=base.shape)
    centers = base + jitter
    radii = 0.40 + rng.normal(scale=0.025, size=len(base))
    obstacles = [Obstacle(c, float(max(0.34, r))) for c, r in zip(centers, radii)]
    return Scenario(
        name="dense",
        bounds=(-0.5, 11.5, -1.55, 1.55),
        start=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        goal=np.array([11.0, 0.0]),
        obstacles=obstacles,
    )


def make_narrow(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    gap_shift = rng.normal(scale=0.035)
    obstacles = []
    for y in (0.56, 1.04, 1.52):
        obstacles.append(Obstacle(np.array([5.35, y + gap_shift]), 0.26))
        obstacles.append(Obstacle(np.array([5.35, -y + gap_shift]), 0.26))
    obstacles.extend(
        [
            Obstacle(np.array([4.35, 1.20]), 0.24),
            Obstacle(np.array([6.35, -1.20]), 0.24),
        ]
    )
    return Scenario(
        name="narrow",
        bounds=(-0.5, 10.5, -1.7, 1.7),
        start=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        goal=np.array([10.0, 0.0]),
        obstacles=obstacles,
    )


def make_forest(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles = []
    xs = np.linspace(1.5, 10.5, 10)
    for x in xs:
        for y in (-0.75, 0.75):
            if rng.random() < 0.82:
                c = np.array([x + rng.normal(scale=0.16), y + rng.normal(scale=0.20)])
                r = 0.22 + 0.06 * rng.random()
                obstacles.append(Obstacle(c, float(r)))
    return Scenario(
        name="forest",
        bounds=(-0.5, 12.5, -1.8, 1.8),
        start=np.array([0.0, -0.1, 0.0, 0.0, 0.0, 0.0]),
        goal=np.array([12.0, 0.1]),
        obstacles=obstacles,
    )


PAPER_DENSE_FOREST_DENSITIES = {
    "paper_dense_forest_d1": 6,
    "paper_dense_forest_d2": 8,
    "paper_dense_forest_d3": 10,
    "paper_dense_forest_d4": 12,
}


def _make_paper_dense_forest(seed: int, tree_count: int, name: str, manual_guide: bool = False) -> Scenario:
    """3 x 20 m tree-corridor benchmark used for paper-style figures.

    The collision model uses tree crowns as conservative trunk envelopes, while
    the visual layer also draws trunks and translucent crowns so the top view
    reads as a forest rather than a collection of gray disks.
    """
    rng = np.random.default_rng(seed)
    obstacles: List[Obstacle] = []
    visuals: List[VisualObject] = []
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([0.0, 20.0])
    y_slots = np.linspace(2.0, 18.0, tree_count)
    signs = np.where(np.arange(tree_count) % 2 == 0, 1.0, -1.0)
    signs[2::5] *= -1.0
    signs[4::6] *= -1.0
    for i, y in enumerate(y_slots):
        lane_bias = 0.44 + 0.05 * np.sin(0.7 * i + 0.4 * tree_count)
        x = lane_bias * signs[i] + rng.normal(scale=0.10)
        center = np.array([float(np.clip(x, -1.02, 1.02)), float(y + rng.normal(scale=0.18))])
        trunk = float(0.20 + 0.04 * rng.random())
        crown = float(0.40 + 0.07 * rng.random())
        height = float(2.0 + 0.5 * rng.random())
        _add_tree(obstacles, visuals, center, trunk, trunk, height)
        visuals.append(VisualObject("tree_crown", center, (2.0 * crown, 2.0 * crown, 0.80), color=(0.12, 0.38, 0.20, 0.42)))

    guide = None
    if manual_guide:
        guide = np.array(
            [
                [0.0, 0.0],
                [-0.52, 2.6],
                [0.58, 4.6],
                [0.44, 6.4],
                [-0.55, 8.5],
                [-0.42, 10.5],
                [0.55, 12.8],
                [-0.52, 15.2],
                [0.38, 17.6],
                [0.0, 20.0],
            ],
            dtype=float,
        )
    return Scenario(
        name=name,
        bounds=(-1.5, 1.5, -0.8, 20.8),
        start=start,
        goal=goal,
        obstacles=obstacles,
        path=guide,
        visuals=visuals,
    )


def make_paper_dense_forest(seed: int) -> Scenario:
    return _make_paper_dense_forest(seed, tree_count=10, name="paper_dense_forest", manual_guide=True)


def make_paper_dense_forest_density(seed: int, name: str) -> Scenario:
    return _make_paper_dense_forest(seed, tree_count=PAPER_DENSE_FOREST_DENSITIES[name], name=name, manual_guide=False)


def make_paper_narrow_gap(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles: List[Obstacle] = []
    visuals: List[VisualObject] = []
    start = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([10.0, 0.0])
    gap = float(0.46 + rng.normal(scale=0.018))
    gap_center = float(0.13 + rng.normal(scale=0.020))
    wall_radius = 0.16
    gate_x = 5.0 + rng.normal(scale=0.04)
    for sign in (-1.0, 1.0):
        y0 = gap_center + sign * (0.5 * gap + wall_radius)
        y_end = gap_center + sign * 1.35
        for y in np.linspace(y0, y_end, 6):
            obstacles.append(Obstacle(np.array([gate_x + rng.normal(scale=0.015), y]), wall_radius))
        visuals.append(
            VisualObject(
                "wall",
                np.array([gate_x, gap_center + sign * (0.5 * gap + 0.62)]),
                (0.32, 1.25, 1.25),
                yaw=np.pi / 2.0,
                color=(0.42, 0.43, 0.45, 0.86),
            )
        )

    for cx, cy, yaw in ((3.2, 0.88, -0.25), (6.8, -0.86, 0.28)):
        _add_wall(
            obstacles,
            visuals,
            np.array([cx + rng.normal(scale=0.04), cy + rng.normal(scale=0.04)]),
            length=1.15,
            thickness=0.18,
            yaw=yaw,
            height=1.0,
            color=(0.35, 0.37, 0.40, 0.70),
        )

    return Scenario(
        name="paper_narrow_gap",
        bounds=(-0.6, 10.6, -1.55, 1.55),
        start=start,
        goal=goal,
        obstacles=obstacles,
        path=np.vstack((start[:2], goal)),
        visuals=visuals,
    )


def _add_tree(
    obstacles: List[Obstacle],
    visuals: List[VisualObject],
    center: np.ndarray,
    trunk_radius: float,
    crown_radius: float,
    height: float,
) -> None:
    obstacles.append(Obstacle(center, crown_radius))
    visuals.append(VisualObject("tree_trunk", center, (2.0 * trunk_radius, 2.0 * trunk_radius, height), color=(0.38, 0.23, 0.12, 1.0)))
    visuals.append(VisualObject("tree_crown", center + np.array([0.0, 0.0]), (2.0 * crown_radius, 2.0 * crown_radius, 0.85), color=(0.18, 0.45, 0.22, 0.72)))


def _add_wall(
    obstacles: List[Obstacle],
    visuals: List[VisualObject],
    center: np.ndarray,
    length: float,
    thickness: float,
    yaw: float,
    height: float,
    color: Tuple[float, float, float, float],
) -> None:
    step = max(0.18, thickness * 0.9)
    count = max(2, int(np.ceil(length / step)))
    direction = np.array([np.cos(yaw), np.sin(yaw)])
    for t in np.linspace(-0.5 * length, 0.5 * length, count):
        obstacles.append(Obstacle(center + t * direction, 0.5 * thickness))
    visuals.append(VisualObject("wall", center, (length, thickness, height), yaw=yaw, color=color))


def _add_block(
    obstacles: List[Obstacle],
    visuals: List[VisualObject],
    center: np.ndarray,
    size_xy: Tuple[float, float],
    yaw: float,
    height: float,
    color: Tuple[float, float, float, float],
) -> None:
    lx, ly = size_xy
    nx = max(2, int(np.ceil(lx / 0.28)))
    ny = max(2, int(np.ceil(ly / 0.28)))
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    for x in np.linspace(-0.5 * lx, 0.5 * lx, nx):
        for y in np.linspace(-0.5 * ly, 0.5 * ly, ny):
            obstacles.append(Obstacle(center + rot @ np.array([x, y]), 0.16))
    visuals.append(VisualObject("block", center, (lx, ly, height), yaw=yaw, color=color))


def make_super_forest(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles: List[Obstacle] = []
    visuals: List[VisualObject] = []
    start = np.array([0.0, -0.15, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([16.0, 0.18])
    xs = np.linspace(1.4, 15.0, 14)
    for i, x in enumerate(xs):
        corridor = 0.30 * np.sin(0.75 * x)
        rows = [-1.55, -0.95, 0.95, 1.55]
        for row in rows:
            if rng.random() < 0.88:
                y = row + corridor + rng.normal(scale=0.16)
                c = np.array([x + rng.normal(scale=0.18), y])
                trunk = 0.08 + 0.03 * rng.random()
                crown = 0.24 + 0.08 * rng.random()
                height = 1.6 + 0.7 * rng.random()
                _add_tree(obstacles, visuals, c, trunk, crown, height)
    for x in (4.8, 8.9, 12.2):
        _add_wall(
            obstacles,
            visuals,
            np.array([x, 0.34 * np.sin(x)]),
            length=1.7,
            thickness=0.18,
            yaw=0.35 * np.sin(x),
            height=1.0,
            color=(0.42, 0.36, 0.30, 0.82),
        )
    return Scenario(
        name="super_forest",
        bounds=(-0.8, 16.8, -2.4, 2.4),
        start=start,
        goal=goal,
        obstacles=obstacles,
        visuals=visuals,
    )


def make_ruin(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles: List[Obstacle] = []
    visuals: List[VisualObject] = []
    start = np.array([0.0, -0.35, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([14.0, 0.25])
    wall_color = (0.50, 0.50, 0.48, 0.86)
    debris_color = (0.36, 0.34, 0.32, 0.86)
    _add_wall(obstacles, visuals, np.array([2.6, 0.90]), 2.4, 0.20, 0.10, 1.25, wall_color)
    _add_wall(obstacles, visuals, np.array([4.3, -0.78]), 2.1, 0.20, -0.45, 1.15, wall_color)
    _add_wall(obstacles, visuals, np.array([6.2, 0.74]), 2.7, 0.22, 0.42, 1.35, wall_color)
    _add_wall(obstacles, visuals, np.array([8.4, -0.62]), 2.4, 0.24, -0.22, 1.10, wall_color)
    _add_wall(obstacles, visuals, np.array([10.5, 0.82]), 2.8, 0.22, 0.35, 1.30, wall_color)
    _add_wall(obstacles, visuals, np.array([12.0, -0.75]), 2.0, 0.22, -0.55, 1.0, wall_color)
    for _ in range(26):
        c = np.array([rng.uniform(1.4, 13.2), rng.uniform(-1.55, 1.55)])
        if abs(c[1]) < 0.28 and 4.8 < c[0] < 9.5:
            continue
        size = (rng.uniform(0.24, 0.55), rng.uniform(0.18, 0.48))
        _add_block(obstacles, visuals, c, size, rng.uniform(-1.2, 1.2), rng.uniform(0.25, 0.85), debris_color)
    return Scenario(
        name="ruin",
        bounds=(-0.8, 14.8, -2.1, 2.1),
        start=start,
        goal=goal,
        obstacles=obstacles,
        visuals=visuals,
    )


def make_warehouse(seed: int) -> Scenario:
    rng = np.random.default_rng(seed)
    obstacles: List[Obstacle] = []
    visuals: List[VisualObject] = []
    start = np.array([0.0, -1.20, 0.0, 0.0, 0.0, 0.0])
    goal = np.array([15.5, 1.10])
    rack_color = (0.26, 0.32, 0.42, 0.82)
    box_color = (0.72, 0.48, 0.26, 0.80)
    for x in np.linspace(2.0, 13.5, 7):
        for y in (-1.45, 1.45):
            _add_wall(obstacles, visuals, np.array([x, y]), 1.25, 0.18, np.pi / 2.0, 1.4, rack_color)
    for _ in range(18):
        c = np.array([rng.uniform(1.3, 14.2), rng.choice([-0.72, 0.72]) + rng.normal(scale=0.10)])
        _add_block(obstacles, visuals, c, (rng.uniform(0.30, 0.55), rng.uniform(0.28, 0.52)), rng.uniform(-0.4, 0.4), rng.uniform(0.35, 0.95), box_color)
    return Scenario(
        name="warehouse",
        bounds=(-0.8, 16.2, -2.2, 2.2),
        start=start,
        goal=goal,
        obstacles=obstacles,
        visuals=visuals,
    )


def build_scenario(name: str, seed: int) -> Scenario:
    if name in PAPER_DENSE_FOREST_DENSITIES:
        return make_paper_dense_forest_density(seed, name)
    if name == "paper_dense_forest":
        return make_paper_dense_forest(seed)
    if name == "paper_narrow_gap":
        return make_paper_narrow_gap(seed)
    if name == "dense":
        return make_dense(seed)
    if name == "narrow":
        return make_narrow(seed)
    if name == "forest":
        return make_forest(seed)
    if name == "super_forest":
        return make_super_forest(seed)
    if name == "ruin":
        return make_ruin(seed)
    if name == "warehouse":
        return make_warehouse(seed)
    raise ValueError("Unknown scenario: %s" % name)


def _point_to_cell(point: Sequence[float], bounds: Tuple[float, float, float, float], res: float) -> Tuple[int, int]:
    xmin, _, ymin, _ = bounds
    return (int(round((point[0] - xmin) / res)), int(round((point[1] - ymin) / res)))


def _cell_to_point(cell: Tuple[int, int], bounds: Tuple[float, float, float, float], res: float) -> np.ndarray:
    xmin, _, ymin, _ = bounds
    return np.array([xmin + cell[0] * res, ymin + cell[1] * res], dtype=float)


def plan_grid_path(scenario: Scenario, res: float, margin: float) -> np.ndarray:
    xmin, xmax, ymin, ymax = scenario.bounds
    nx = int(round((xmax - xmin) / res)) + 1
    ny = int(round((ymax - ymin) / res)) + 1
    start = _point_to_cell(scenario.start[:2], scenario.bounds, res)
    goal = _point_to_cell(scenario.goal, scenario.bounds, res)
    occupied = np.zeros((nx, ny), dtype=bool)
    for ix in range(nx):
        for iy in range(ny):
            p = _cell_to_point((ix, iy), scenario.bounds, res)
            for obs in scenario.obstacles:
                if np.linalg.norm(p - obs.center) <= obs.radius + margin:
                    occupied[ix, iy] = True
                    break
    for cell in (start, goal):
        if 0 <= cell[0] < nx and 0 <= cell[1] < ny:
            occupied[cell[0], cell[1]] = False

    moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    pq = [(0.0, start)]
    came = {}
    gscore = {start: 0.0}
    closed = set()

    def h(cell):
        return res * np.linalg.norm(np.array(cell) - np.array(goal))

    while pq:
        _, cell = heapq.heappop(pq)
        if cell in closed:
            continue
        if cell == goal:
            break
        closed.add(cell)
        for dx, dy in moves:
            nb = (cell[0] + dx, cell[1] + dy)
            if nb[0] < 0 or nb[0] >= nx or nb[1] < 0 or nb[1] >= ny or occupied[nb]:
                continue
            step = res * (2.0 ** 0.5 if dx and dy else 1.0)
            ng = gscore[cell] + step
            if ng < gscore.get(nb, float("inf")):
                gscore[nb] = ng
                came[nb] = cell
                heapq.heappush(pq, (ng + h(nb), nb))

    if goal not in came:
        return np.vstack((scenario.start[:2], scenario.goal))

    cells = [goal]
    while cells[-1] != start:
        cells.append(came[cells[-1]])
    cells.reverse()
    path = np.vstack([_cell_to_point(c, scenario.bounds, res) for c in cells])
    path[0] = scenario.start[:2]
    path[-1] = scenario.goal
    return simplify_path(path, scenario.obstacles, margin)


def _line_is_free(a: np.ndarray, b: np.ndarray, obstacles: List[Obstacle], margin: float) -> bool:
    seg = b - a
    denom = float(np.dot(seg, seg))
    for obs in obstacles:
        if denom < 1e-12:
            dist = np.linalg.norm(a - obs.center)
        else:
            t = float(np.clip(np.dot(obs.center - a, seg) / denom, 0.0, 1.0))
            closest = a + t * seg
            dist = np.linalg.norm(closest - obs.center)
        if dist <= obs.radius + margin:
            return False
    return True


def simplify_path(path: np.ndarray, obstacles: List[Obstacle], margin: float) -> np.ndarray:
    if len(path) <= 2:
        return path
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_is_free(path[i], path[j], obstacles, margin):
            j -= 1
        simplified.append(path[j])
        i = j
    return np.vstack(simplified)


def reference_from_path(pos: np.ndarray, path: np.ndarray, goal: np.ndarray, lookahead: float) -> np.ndarray:
    if path is None or len(path) == 0:
        return goal
    best_s = 0.0
    best_d = float("inf")
    accum = 0.0
    seg_lengths = []
    for j in range(len(path) - 1):
        seg = path[j + 1] - path[j]
        seg_len = float(np.linalg.norm(seg))
        seg_lengths.append(seg_len)
        if seg_len > 1e-9:
            t = float(np.clip(np.dot(pos - path[j], seg) / (seg_len * seg_len), 0.0, 1.0))
            proj = path[j] + t * seg
            dist = float(np.linalg.norm(pos - proj))
            if dist < best_d:
                best_d = dist
                best_s = accum + t * seg_len
        accum += seg_len
    target_s = best_s + lookahead
    accum = 0.0
    for j, seg in enumerate(seg_lengths):
        if accum + seg >= target_s:
            t = (target_s - accum) / max(seg, 1e-9)
            return (1.0 - t) * path[j] + t * path[j + 1]
        accum += seg
    return goal
