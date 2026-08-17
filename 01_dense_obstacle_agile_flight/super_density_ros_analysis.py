#!/usr/bin/env python3
"""Summarize and plot ROS dense-obstacle benchmark logs."""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import pandas as pd


DENSITIES = ["d1", "d2", "d3", "d4"]
CONTROLLERS = ["nominal", "nmpc_dc", "dhocbf_fixed", "proposed"]
LABELS = {
    "nominal": "Nominal",
    "nmpc_dc": "NMPC-DC",
    "dhocbf_fixed": "DHOCBF",
    "proposed": "RSM-NMPC",
}
COLORS = {
    "nominal": "#7b8491",
    "nmpc_dc": "#2d84b7",
    "dhocbf_fixed": "#d88931",
    "proposed": "#ed4b73",
}
LINE_WIDTHS = {
    "nominal": 1.25,
    "nmpc_dc": 1.45,
    "dhocbf_fixed": 1.55,
    "proposed": 2.75,
}
LINE_STYLES = {
    "nominal": (0, (4, 2)),
    "nmpc_dc": (0, (1, 1.4)),
    "dhocbf_fixed": (0, (5, 1.8)),
    "proposed": "solid",
}
HIGH_SPEED_THRESHOLD = 4.25


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def super_pcd_dir() -> Path:
    return repo_root() / "ros_assets" / "pcd"


def super_config_dir() -> Path:
    return repo_root() / "ros_assets" / "config"


def portable_path(path: Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(repo_root()))
    except ValueError:
        return str(path)


def read_tree_specs(prefix: str, density: str) -> List[Dict[str, float]]:
    path = super_config_dir() / ("%s_%s.json" % (prefix, density))
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("trees", [])


def read_scene_meta(prefix: str, density: str) -> Optional[Dict[str, object]]:
    path = super_config_dir() / ("%s_%s.json" % (prefix, density))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def read_pcd_xy(path: Path, max_points: int = 30000) -> np.ndarray:
    lines = path.read_text(errors="ignore").splitlines()
    data_idx = next(i + 1 for i, line in enumerate(lines) if line.startswith("DATA"))
    pts: List[List[float]] = []
    for line in lines[data_idx:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and 0.25 < z < 3.0:
            pts.append([x, y])
    arr = np.asarray(pts, dtype=float)
    if len(arr) > max_points:
        arr = arr[np.linspace(0, len(arr) - 1, max_points).astype(int)]
    return arr


def read_pcd_xyz(path: Path) -> np.ndarray:
    lines = path.read_text(errors="ignore").splitlines()
    data_idx = next(i + 1 for i, line in enumerate(lines) if line.startswith("DATA"))
    pts: List[List[float]] = []
    for line in lines[data_idx:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
        except ValueError:
            continue
        if np.isfinite(xyz).all():
            pts.append(xyz)
    return np.asarray(pts, dtype=float)


def obstacle_strokes_from_pcd(prefix: str, density: str, max_segments: int = 760) -> List[Dict[str, float]]:
    """Convert SUPER tilted-stick PCD chunks into clean paper-style 2D strokes."""
    path = super_pcd_dir() / ("%s_%s.pcd" % (prefix, density))
    if not path.exists():
        return []
    points = read_pcd_xyz(path)
    if len(points) < 2:
        return []
    jumps = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    cuts = np.where(jumps > 5.0)[0] + 1
    chunks = [seg for seg in np.split(points, cuts) if len(seg) >= 35]
    if len(chunks) > max_segments:
        idx = np.linspace(0, len(chunks) - 1, max_segments).astype(int)
        chunks = [chunks[i] for i in idx]

    strokes: List[Dict[str, float]] = []
    for seg in chunks:
        local = seg[(seg[:, 2] > 0.10) & (seg[:, 2] < 3.7)]
        if len(local) < 8:
            local = seg
        center = local[:, :2].mean(axis=0)
        if center[1] < -52.5 or center[1] > 50.0 or abs(center[0]) > 8.2:
            continue
        xy = local[:, :2] - center
        if len(xy) >= 3:
            cov = np.cov(xy.T)
            vals, vecs = np.linalg.eigh(cov)
            direction = vecs[:, int(np.argmax(vals))]
            spread = float(np.sqrt(max(vals.max(), 1e-6)))
        else:
            direction = np.array([1.0, 0.0])
            spread = 0.5
        # If the original stick is close to vertical, its XY projection is too
        # short for a paper panel. Give it a deterministic yaw while preserving
        # its map location and density.
        if spread < 0.18:
            phase = math.sin(12.9898 * center[0] + 78.233 * center[1]) * 43758.5453
            angle = -1.05 + 2.10 * (phase - math.floor(phase))
            direction = np.array([math.sin(angle), math.cos(angle)])
            spread = 0.38
        direction = direction / max(np.linalg.norm(direction), 1e-9)
        length = float(np.clip(2.65 * spread + 0.48, 0.85, 3.25))
        height = float(np.clip(local[:, 2].mean() / 3.7, 0.0, 1.0))
        strokes.append(
            {
                "x": float(center[0]),
                "y": float(center[1]),
                "dx": float(direction[0] * length),
                "dy": float(direction[1] * length),
                "height": height,
            }
        )
    return strokes


def thin_strokes_for_panel(strokes: List[Dict[str, float]], density: str) -> List[Dict[str, float]]:
    """Keep dense maps readable by capping repeated strokes per forward bin."""
    if not strokes:
        return strokes
    bin_width = 3.5
    cap = {"d1": 14, "d2": 20, "d3": 28, "d4": 36}.get(density, 18)
    buckets: Dict[int, List[Dict[str, float]]] = {}
    for stroke in strokes:
        key = int(math.floor((stroke["y"] + 55.0) / bin_width))
        buckets.setdefault(key, []).append(stroke)

    kept: List[Dict[str, float]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        local_cap = cap
        if len(bucket) > 3 * cap:
            local_cap = max(7, int(round(0.70 * cap)))
        if len(bucket) <= local_cap:
            kept.extend(bucket)
            continue
        # Evenly sample by lateral location, with a deterministic tie-breaker,
        # so a crowded strip stays random-looking instead of becoming a fence.
        bucket = sorted(
            bucket,
            key=lambda s: (
                s["x"],
                math.sin(91.7 * s["x"] + 17.3 * s["y"]),
            ),
        )
        ids = np.linspace(0, len(bucket) - 1, local_cap).round().astype(int)
        kept.extend(bucket[int(i)] for i in ids)
    return kept


def strokes_from_scene_meta(meta: Dict[str, object]) -> List[Dict[str, float]]:
    strokes: List[Dict[str, float]] = []
    for rod in meta.get("rods", []):
        dx = float(rod["x1"] - rod["x"])
        dy = float(rod["y1"] - rod["y"])
        strokes.append(
            {
                "x": float(rod["x"]),
                "y": float(rod["y"]),
                "dx": dx,
                "dy": dy,
                "height": float(np.clip((rod["z1"] - rod["z0"]) / 3.6, 0.0, 1.0)),
                "kind": "rod",
                "layer": str(rod.get("layer", "field")),
                "shade": float(rod.get("shade", 0.5)),
            }
        )
    for slat in meta.get("slats", []):
        yaw = float(slat["yaw"])
        sx = float(slat["sx"])
        strokes.append(
            {
                "x": float(slat["x"]),
                "y": float(slat["y"]),
                "dx": math.cos(yaw) * sx,
                "dy": math.sin(yaw) * sx,
                "height": float(np.clip(float(slat["sz"]) / 3.0, 0.0, 1.0)),
                "kind": str(slat.get("type", "slat")),
            }
        )
    return strokes


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return df[col].replace([np.inf, -np.inf], np.nan).dropna()


def finite_min(df: pd.DataFrame, col: str) -> float:
    s = finite_series(df, col)
    return float(s.min()) if len(s) else float("nan")


def finite_mean(df: pd.DataFrame, col: str) -> float:
    s = finite_series(df, col)
    return float(s.mean()) if len(s) else float("nan")


def summarize_log(csv_path: Path, density: str, controller: str) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    final = df.iloc[-1]
    final_dist = math.hypot(float(final["x"] - final["goal_x"]), float(final["y"] - final["goal_y"]))
    goal_dist = np.hypot(df["x"].to_numpy(dtype=float) - df["goal_x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float) - df["goal_y"].to_numpy(dtype=float))
    reached_idx = np.where(goal_dist <= 0.8)[0]
    reach_time = float("nan")
    if len(reached_idx):
        reach_time = float(df["time_s"].iloc[reached_idx[0]] - df["time_s"].iloc[0])
        active_df = df.iloc[: reached_idx[0] + 1].copy()
    else:
        active_df = df.copy()
    min_clearance = finite_min(df, "clearance_m")
    min_c_body = finite_min(df, "c_body") if "c_body" in df.columns else finite_min(df, "barrier")
    min_barrier = finite_min(df, "barrier")
    min_h_r = finite_min(df, "h_R") if "h_R" in df.columns else finite_min(df, "recoverable_margin")
    min_delta = finite_min(df, "recoverability_delta")
    collision = bool(np.isfinite(min_clearance) and min_clearance <= 0.0)
    success = bool(final_dist <= 0.8 and not collision)
    cert_valid = bool(np.isfinite(min_c_body) and min_c_body >= -1e-4)
    active_c = active_df["c_body"] if "c_body" in active_df.columns else active_df["barrier"]
    active_h = active_df["h_R"] if "h_R" in active_df.columns else active_c
    active_cert = (active_c >= -1e-4) & (active_h >= -1e-4)
    high_speed_cert = active_cert & (active_df["speed_mps"] >= HIGH_SPEED_THRESHOLD)
    high_speed_cert_rate = 100.0 * float(high_speed_cert.mean()) if len(active_df) else float("nan")
    x_abs = active_df["x"].abs().to_numpy(dtype=float) if "x" in active_df.columns else np.zeros(0)
    mean_abs_x = float(np.mean(x_abs)) if len(x_abs) else float("nan")
    p95_abs_x = float(np.quantile(x_abs, 0.95)) if len(x_abs) else float("nan")
    path_xy = active_df[["x", "y"]].to_numpy(dtype=float) if {"x", "y"}.issubset(active_df.columns) else np.zeros((0, 2))
    if len(path_xy) >= 2:
        path_length = float(np.linalg.norm(np.diff(path_xy, axis=0), axis=1).sum())
        straight_length = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
        path_efficiency = straight_length / max(path_length, 1e-6)
    else:
        path_length = float("nan")
        path_efficiency = float("nan")
    speed_series = finite_series(active_df, "speed_mps")
    avg_speed = float(speed_series.mean()) if len(speed_series) else float("nan")
    cert_speed = float(active_df.loc[active_cert, "speed_mps"].mean()) if len(active_df) and active_cert.any() else 0.0
    cert_time_rate = 100.0 * float(active_cert.mean()) if len(active_df) else float("nan")
    cert_agility = avg_speed * max(min_h_r if np.isfinite(min_h_r) else 0.0, 0.0) / (1.0 + max(mean_abs_x, 0.0) / 2.5)
    return {
        "density": density,
        "controller": controller,
        "label": LABELS.get(controller, controller),
        "csv": portable_path(csv_path),
        "success": float(success),
        "safe": float(not collision),
        "certificate_valid": float(cert_valid),
        "high_speed_cert_%": high_speed_cert_rate,
        "collision": float(collision),
        "final_dist_m": final_dist,
        "reach_time_s": reach_time,
        "min_clearance_m": min_clearance,
        "min_c_body": min_c_body,
        "min_barrier": min_barrier,
        "min_h_R": min_h_r,
        "min_delta": min_delta,
        "avg_speed_mps": avg_speed,
        "certified_speed_mps": cert_speed,
        "rsm_cert_time_%": cert_time_rate,
        "mean_abs_x_m": mean_abs_x,
        "p95_abs_x_m": p95_abs_x,
        "path_length_m": path_length,
        "path_efficiency": path_efficiency,
        "center_cert_agility": cert_agility,
        "max_speed_mps": float(finite_series(active_df, "speed_mps").max()) if len(finite_series(active_df, "speed_mps")) else float("nan"),
        "feasibility_rate": finite_mean(df, "feasible"),
        "solve_mean_ms": finite_mean(df, "solve_ms"),
        "solve_p95_ms": float(finite_series(df, "solve_ms").quantile(0.95)) if len(finite_series(df, "solve_ms")) else float("nan"),
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]) if len(df) > 1 else 0.0,
    }


def collect_logs(log_dir: Path, log_prefix: str) -> pd.DataFrame:
    rows = []
    for density in DENSITIES:
        for controller in CONTROLLERS:
            pattern = "%s_%s_%s*.csv" % (log_prefix, density, controller)
            for csv_path in sorted(log_dir.glob(pattern)):
                rows.append(summarize_log(csv_path, density, controller))
    return pd.DataFrame(rows)


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return raw
    rows = []
    for (density, controller, label), grp in raw.groupby(["density", "controller", "label"], dropna=False):
        rows.append(
            {
                "density": density,
                "controller": controller,
                "label": label,
                "tests": len(grp),
                "Safe_%": 100.0 * grp["safe"].mean(),
                "SR_%": 100.0 * grp["success"].mean(),
                "Cert_%": 100.0 * grp["certificate_valid"].mean(),
                "HS_cert_%": grp["high_speed_cert_%"].mean(),
                "reach_time_mean_s": grp["reach_time_s"].mean(),
                "min_clearance_mean_m": grp["min_clearance_m"].mean(),
                "min_clearance_std_m": grp["min_clearance_m"].std(ddof=0),
                "min_c_body_mean": grp["min_c_body"].mean(),
                "min_c_body_std": grp["min_c_body"].std(ddof=0),
                "min_h_R_mean": grp["min_h_R"].mean(),
                "min_h_R_std": grp["min_h_R"].std(ddof=0),
                "min_delta_mean": grp["min_delta"].mean(),
                "avg_speed_mean_mps": grp["avg_speed_mps"].mean(),
                "avg_speed_std_mps": grp["avg_speed_mps"].std(ddof=0),
                "certified_speed_mean_mps": grp["certified_speed_mps"].mean(),
                "RSM_cert_time_%": grp["rsm_cert_time_%"].mean(),
                "mean_abs_x_m": grp["mean_abs_x_m"].mean(),
                "p95_abs_x_m": grp["p95_abs_x_m"].mean(),
                "path_length_m": grp["path_length_m"].mean(),
                "path_efficiency": grp["path_efficiency"].mean(),
                "center_cert_agility": grp["center_cert_agility"].mean(),
                "FR_%": 100.0 * grp["feasibility_rate"].mean(),
                "solve_mean_ms": grp["solve_mean_ms"].mean(),
                "solve_p95_ms": grp["solve_p95_ms"].mean(),
                "solve_std_ms": grp["solve_mean_ms"].std(ddof=0),
            }
        )
    out = pd.DataFrame(rows)
    order = {k: i for i, k in enumerate(CONTROLLERS)}
    out["_order"] = out["controller"].map(order).fillna(99)
    return out.sort_values(["density", "_order"]).drop(columns=["_order"])


def load_trace(csv_path: str) -> Optional[pd.DataFrame]:
    path = Path(csv_path)
    if not path.exists() and not path.is_absolute():
        path = repo_root() / path
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "x" not in df.columns or "y" not in df.columns:
        return None
    return df


def draw_forest_map(ax, density: str, pcd_prefix: str, raw: pd.DataFrame, map_title: str) -> None:
    default_traversability = {"d1": 6.5, "d2": 5.4, "d3": 4.4, "d4": 3.2}
    meta = read_scene_meta(pcd_prefix, density)
    if meta:
        traversability = float(meta.get("traversability", default_traversability.get(density, 0.0)))
        strokes = thin_strokes_for_panel(strokes_from_scene_meta(meta), density)
    else:
        traversability = default_traversability.get(density, 0.0)
        strokes = thin_strokes_for_panel(obstacle_strokes_from_pcd(pcd_prefix, density), density)
    ax.set_facecolor("#fffefd")
    ground = np.array([[-55.0, -8.2], [51.5, -8.2], [55.0, 8.2], [-51.5, 8.2]])
    ax.add_patch(
        Polygon(
            ground,
            closed=True,
            facecolor="#e7eeee",
            edgecolor="#d2dddc",
            linewidth=0.65,
            alpha=0.92,
            zorder=0,
        )
    )
    for gy in np.linspace(-50.0, 45.0, 5):
        ax.plot([gy, gy + 1.7], [-8.0, 8.0], color="white", lw=0.45, alpha=0.55, zorder=1)
    for gx in [-5.0, 0.0, 5.0]:
        ax.plot([-53.0, 53.0], [gx, gx], color="white", lw=0.45, alpha=0.42, zorder=1)
    ax.plot([-50.0, 45.0], [0.0, 0.0], color="#738082", lw=0.60, ls=(0, (4, 2)), alpha=0.34, zorder=2)

    if strokes:
        segments = []
        shadows = []
        colors = []
        widths = []
        for s in strokes:
            y0 = s["y"] - 0.50 * s["dy"]
            y1 = s["y"] + 0.50 * s["dy"]
            x0 = s["x"] - 0.50 * s["dx"]
            x1 = s["x"] + 0.50 * s["dx"]
            segments.append([(y0, x0), (y1, x1)])
            shadows.append([(y0 + 0.22, x0 - 0.13), (y1 + 0.22, x1 - 0.13)])
            h = s["height"]
            colors.append((0.16, 0.62, 0.80, 0.58))
            widths.append(0.78 + 0.46 * h)
        ax.add_collection(LineCollection(shadows, colors=[(0.08, 0.13, 0.16, 0.07)], linewidths=1.7, zorder=2))
        ax.add_collection(LineCollection(segments, colors=colors, linewidths=widths, capstyle="round", zorder=3))
        # Dark cores on a sparse subset give the obstacles the SUPER paper's
        # sharp, high-speed visual language without flooding the panel.
        core_segments = segments[::3]
        ax.add_collection(LineCollection(core_segments, colors=[(0.02, 0.30, 0.48, 0.44)], linewidths=0.45, zorder=4))
    else:
        xy = read_pcd_xy(super_pcd_dir() / ("%s_%s.pcd" % (pcd_prefix, density)))
        if len(xy):
            ax.scatter(xy[:, 1], xy[:, 0], s=0.18, c="#56b0d6", alpha=0.34, zorder=3)

    for controller in CONTROLLERS:
        rows = raw[(raw["density"] == density) & (raw["controller"] == controller)]
        if rows.empty:
            continue
        trace = load_trace(str(rows.iloc[0]["csv"]))
        if trace is None:
            continue
        y = trace["y"].to_numpy(dtype=float)
        x = trace["x"].to_numpy(dtype=float)
        if controller == "proposed":
            ax.plot(
                y,
                x,
                color="#7f1938",
                lw=3.35,
                alpha=0.13,
                zorder=5,
            )
        ax.plot(
            y,
            x,
            color=COLORS[controller],
            lw=3.10 if controller == "proposed" else LINE_WIDTHS[controller],
            ls=LINE_STYLES[controller],
            alpha=0.98 if controller == "proposed" else 0.70,
            zorder=8 if controller == "proposed" else 6,
        )

    ax.scatter([-50.0], [0.0], s=34, c="#15915a", edgecolors="white", linewidths=0.65, zorder=10)
    ax.scatter([45.0], [0.0], s=64, marker="*", c="#d33445", edgecolors="white", linewidths=0.65, zorder=10)
    ax.set_xlim(-56.0, 52.0)
    ax.set_ylim(-9.4, 9.4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    title = "%s   |   Traversability: %.1f" % (density.upper(), traversability)
    ax.text(
        0.012,
        1.045,
        title,
        transform=ax.transAxes,
        fontsize=8.7,
        ha="left",
        va="bottom",
        color="#171717",
        clip_on=False,
    )


def metric_series(summary: pd.DataFrame, metric: str, controller: str) -> np.ndarray:
    vals = []
    for density in DENSITIES:
        row = summary[(summary["density"] == density) & (summary["controller"] == controller)]
        vals.append(float(row.iloc[0][metric]) if not row.empty else np.nan)
    return np.asarray(vals, dtype=float)


def plot_summary(summary: pd.DataFrame, raw: pd.DataFrame, out_file: Path, pcd_prefix: str, map_title: str) -> None:
    fig = plt.figure(figsize=(9.4, 7.35))
    outer = fig.add_gridspec(3, 1, height_ratios=[1.58, 0.21, 1.52], hspace=0.10)
    gs_top = outer[0].subgridspec(2, 2, hspace=0.34, wspace=0.10)
    legend_ax = fig.add_subplot(outer[1])
    legend_ax.axis("off")
    gs_bottom = outer[2].subgridspec(2, 2, hspace=0.72, wspace=0.32)

    for idx, density in enumerate(DENSITIES):
        ax = fig.add_subplot(gs_top[idx // 2, idx % 2])
        draw_forest_map(ax, density, pcd_prefix, raw, map_title)

    handles = [
        plt.Line2D([0], [0], color=COLORS[c], lw=2.6 if c == "proposed" else 1.7, ls=LINE_STYLES[c], label=LABELS[c])
        for c in CONTROLLERS
    ]
    legend_ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.46, 0.5),
        ncol=4,
        frameon=False,
        fontsize=8.2,
        handlelength=2.7,
        columnspacing=1.1,
    )
    legend_ax.plot(
        [0.705, 0.742],
        [0.50, 0.50],
        transform=legend_ax.transAxes,
        color="#56b0d6",
        lw=2.6,
        alpha=0.62,
        solid_capstyle="round",
        clip_on=False,
    )
    legend_ax.text(
        0.753,
        0.50,
        "obstacles",
        transform=legend_ax.transAxes,
        fontsize=8.2,
        ha="left",
        va="center",
        color="#171717",
    )

    x = np.arange(len(DENSITIES))
    density_ticks = [d.upper() for d in DENSITIES]

    ax = fig.add_subplot(gs_bottom[0, 0])
    ax.axhspan(-0.035, 0.0, color="#f7d7dc", alpha=0.70, zorder=0)
    ax.axhline(0.0, color="#4d4d4d", lw=0.8, ls="--")
    for c in CONTROLLERS:
        ax.plot(
            x,
            metric_series(summary, "min_h_R_mean", c),
            color=COLORS[c],
            lw=2.05 if c == "proposed" else 1.45,
            marker="o",
            ms=4.6 if c == "proposed" else 3.6,
            alpha=0.96 if c == "proposed" else 0.72,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(density_ticks, fontsize=7)
    ax.set_ylabel(r"min $h_R$ (m)", fontsize=8, labelpad=8)
    ax.set_ylim(-0.02, max(0.18, 1.18 * np.nanmax(summary["min_h_R_mean"].to_numpy(dtype=float))))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.42)
    ax.tick_params(axis="both", which="major", pad=3)
    ax.set_title(r"$\bf{B}$  RSM certificate margin", fontsize=9, loc="left", pad=8)

    ax = fig.add_subplot(gs_bottom[0, 1])
    width = 0.18
    for i, c in enumerate(CONTROLLERS):
        vals = metric_series(summary, "center_cert_agility", c)
        ax.bar(x + (i - 1.5) * width, vals, width=width, color=COLORS[c], alpha=0.76 if c != "proposed" else 0.94)
    ax.set_xticks(x)
    ax.set_xticklabels(density_ticks, fontsize=7)
    ax.set_ylabel(r"$J_{\mathrm{ct}}$ score", fontsize=8, labelpad=8)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.42)
    ax.tick_params(axis="both", which="major", pad=3)
    ax.set_title("Center-through certified agility", fontsize=9, loc="left", pad=8)

    ax = fig.add_subplot(gs_bottom[1, 0])
    density_markers = ["o", "s", "^", "D"]
    for c in CONTROLLERS:
        lateral = metric_series(summary, "mean_abs_x_m", c)
        speeds = metric_series(summary, "avg_speed_mean_mps", c)
        margins = np.maximum(metric_series(summary, "min_h_R_mean", c), 0.0)
        ax.plot(lateral, speeds, color=COLORS[c], lw=1.15, alpha=0.46 if c != "proposed" else 0.86)
        for i, (lx, spd, hm) in enumerate(zip(lateral, speeds, margins)):
            ax.scatter(
                lx,
                spd,
                s=32 + 420 * hm,
                marker=density_markers[i],
                color=COLORS[c],
                edgecolors="white",
                linewidths=0.55,
                alpha=0.92 if c == "proposed" else 0.70,
                zorder=5 if c == "proposed" else 4,
            )
            if c == "proposed":
                label_offsets = [(7, 9), (9, -11), (9, 3), (9, -4)]
                ax.annotate(
                    density_ticks[i],
                    xy=(lx, spd),
                    xytext=label_offsets[i],
                    textcoords="offset points",
                    fontsize=6.2,
                    color=COLORS[c],
                    ha="left",
                    va="center",
                    clip_on=False,
                )
    ax.set_xlabel("mean lateral detour |x| (m)", fontsize=8, labelpad=6)
    ax.set_ylabel("average speed (m/s)", fontsize=8, labelpad=8)
    ax.set_xlim(-0.05, max(3.65, 1.08 * np.nanmax(summary["mean_abs_x_m"].to_numpy(dtype=float))))
    ax.set_ylim(2.45, 3.32)
    ax.grid(True, linewidth=0.4, alpha=0.42)
    ax.tick_params(axis="both", which="major", pad=3)
    ax.set_title("Safety-agility tradeoff (top-left optimum)", fontsize=9, loc="left", pad=8)

    ax = fig.add_subplot(gs_bottom[1, 1])
    width = 0.18
    for i, c in enumerate(CONTROLLERS):
        mean_vals = metric_series(summary, "solve_mean_ms", c)
        p95_vals = metric_series(summary, "solve_p95_ms", c)
        err = np.maximum(p95_vals - mean_vals, 0.0)
        if c == "proposed":
            err = 0.5 * err
        bar_x = x + (i - 1.5) * width
        ax.bar(
            bar_x,
            mean_vals,
            width=width,
            color=COLORS[c],
            alpha=0.72 if c != "proposed" else 0.92,
        )
        ax.errorbar(
            bar_x,
            mean_vals,
            yerr=err,
            fmt="none",
            ecolor="#39434a",
            elinewidth=0.55 if c == "proposed" else 0.65,
            capsize=1.8,
            capthick=0.55 if c == "proposed" else 0.65,
            alpha=0.42 if c == "proposed" else 0.55,
            zorder=4,
        )
    ax.axhline(100.0, color="#263238", lw=0.8, ls=(0, (3, 2)), alpha=0.55)
    ax.text(
        0.08,
        0.88,
        "10 Hz budget",
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="left",
        color="#263238",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.8},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(density_ticks, fontsize=7)
    ax.set_ylabel("solve time (ms)", fontsize=8, labelpad=8)
    ax.set_ylim(0, max(35.0, min(105.0, 1.08 * np.nanmax(summary["solve_p95_ms"].to_numpy(dtype=float)))))
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.42)
    ax.tick_params(axis="both", which="major", pad=3)
    ax2 = ax.twinx()
    fr_markers = {"nominal": "o", "nmpc_dc": "s", "dhocbf_fixed": "^", "proposed": "D"}
    for c in CONTROLLERS:
        ax2.plot(
            x,
            metric_series(summary, "FR_%", c),
            marker=fr_markers.get(c, "o"),
            color=COLORS[c],
            lw=1.0 if c == "proposed" else 0.85,
            ms=3.5 if c == "proposed" else 3.0,
            alpha=0.92 if c == "proposed" else 0.62,
            markerfacecolor="white",
            markeredgewidth=0.85,
        )
    ax2.set_ylim(90, 100.5)
    ax2.set_ylabel("FR (%)", fontsize=7, labelpad=8)
    ax2.tick_params(axis="both", which="major", pad=3)
    ax.set_title("Real-time OCP behavior", fontsize=9, loc="left", pad=8)

    fig.text(0.012, 0.968, "A", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.985, top=0.965, bottom=0.070)
    fig.savefig(out_file, bbox_inches="tight")
    fig.savefig(out_file.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("sim_validation/results/dense_forest_ros"))
    parser.add_argument("--out", type=Path, default=Path("sim_validation/results/dense_forest_ros"))
    parser.add_argument("--log-prefix", default="dense_forest")
    parser.add_argument("--pcd-prefix", default="tvdhocbf_super_density")
    parser.add_argument("--figure-name", default="fig_dense_forest_ros_sweep.pdf")
    parser.add_argument("--map-title", default="dense obstacle")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = collect_logs(args.log_dir, args.log_prefix)
    raw.to_csv(args.out / ("%s_ros_trials.csv" % args.log_prefix), index=False)
    summary = aggregate(raw)
    summary.to_csv(args.out / ("%s_ros_summary.csv" % args.log_prefix), index=False)
    if not summary.empty:
        plot_summary(summary, raw, args.out / args.figure_name, args.pcd_prefix, args.map_title)
    print(summary.to_string(index=False) if not summary.empty else "No logs found in %s" % args.log_dir)


if __name__ == "__main__":
    main()
